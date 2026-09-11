import json
import os
import uuid

import pytest

from joblens import db
from joblens.models import Posting, RawItem

pytestmark = pytest.mark.db


def _database_available() -> bool:
    try:
        with db.connect() as conn:
            conn.execute("select 1")
        return True
    except Exception:
        return False


pytest.importorskip("psycopg")
if not _database_available():
    pytest.skip(
        "no Postgres reachable at DATABASE_URL, skipping db tests",
        allow_module_level=True,
    )


@pytest.fixture
def conn():
    db.migrate()
    with db.connect() as connection:
        # Each test starts from an empty database. Fine here because this is a
        # throwaway test database; never point DATABASE_URL at anything real.
        connection.execute("truncate postings, raw_postings, ingestion_runs")
        connection.commit()
        yield connection


def make_posting(source_id: str = "1", title: str = "ML Engineer") -> Posting:
    return Posting.build(
        source="remoteok",
        source_id=source_id,
        title=title,
        company="Northwind Analytics",
        url=f"https://example.com/{source_id}",
        location="London, UK",
        salary_raw="£90,000 - £120,000 per year",
        description_html="<p>Build models.</p>",
    )


def test_upsert_inserts_then_updates(conn):
    inserted, updated = db.upsert_postings(conn, [make_posting()])
    assert (inserted, updated) == (1, 0)
    inserted, updated = db.upsert_postings(conn, [make_posting()])
    assert (inserted, updated) == (0, 1)
    total = conn.execute("select count(*) as n from postings").fetchone()["n"]
    assert total == 1


def test_running_twice_does_not_duplicate_rows(conn):
    postings = [make_posting(str(i)) for i in range(5)]
    db.upsert_postings(conn, postings)
    db.upsert_postings(conn, postings)
    total = conn.execute("select count(*) as n from postings").fetchone()["n"]
    assert total == 5


def test_last_seen_at_moves_but_first_seen_at_does_not(conn):
    db.upsert_postings(conn, [make_posting()])
    first = conn.execute("select first_seen_at, last_seen_at from postings").fetchone()
    db.upsert_postings(conn, [make_posting()])
    second = conn.execute("select first_seen_at, last_seen_at from postings").fetchone()
    assert second["first_seen_at"] == first["first_seen_at"]
    assert second["last_seen_at"] >= first["last_seen_at"]


def test_duplicate_count_spots_the_same_job_on_two_boards(conn):
    same_job = Posting.build(
        source="hackernews",
        source_id="99",
        title="Senior ML Engineer",
        company="Northwind Analytics",
        url="https://news.ycombinator.com/item?id=99",
        location="London, UK",
    )
    db.upsert_postings(conn, [make_posting(), same_job])
    assert db.count_duplicates(conn) == 1


def test_raw_rows_round_trip(conn):
    run_id = db.start_run(conn, "remoteok")
    payload = {"id": "42", "position": "ML Engineer", "company": "Acme"}
    item = RawItem(source="remoteok", source_id="42", payload=payload)
    assert db.insert_raw(conn, [item], run_id) == 1
    rows = db.fetch_raw(conn, run_id=run_id)
    assert len(rows) == 1
    stored = rows[0]["payload"]
    if isinstance(stored, str):
        stored = json.loads(stored)
    assert stored["position"] == "ML Engineer"


def test_run_log_records_success(conn):
    run_id = db.start_run(conn, "remoteok")
    db.finish_run(conn, run_id, status="ok", fetched=10, inserted=8, updated=2)
    row = conn.execute(
        "select * from ingestion_runs where run_id = %s", (run_id,)
    ).fetchone()
    assert row["status"] == "ok"
    assert row["fetched"] == 10
    assert row["finished_at"] is not None


def test_run_log_records_failure(conn):
    run_id = db.start_run(conn, "adzuna")
    db.finish_run(conn, run_id, status="failed", error="429 from upstream")
    row = conn.execute(
        "select status, error from ingestion_runs where run_id = %s", (run_id,)
    ).fetchone()
    assert row["status"] == "failed"
    assert "429" in row["error"]


def test_unknown_run_id_returns_nothing(conn):
    assert db.fetch_raw(conn, run_id=uuid.uuid4()) == []


def test_database_url_is_not_a_production_looking_url():
    # Cheap guard against someone running the truncating fixtures against a
    # real database by accident.
    url = os.environ.get("DATABASE_URL", "")
    assert "amazonaws" not in url and "neon.tech" not in url
