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
        connection.execute("truncate postings, raw_postings, ingestion_runs cascade")
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


def _store_vector(conn, posting_id: int, value: float) -> None:
    vector = "[" + ",".join([str(value)] * 384) + "]"
    conn.execute(
        """
        insert into posting_chunks
            (posting_id, strategy, chunk_index, content, model, embedding)
        values (%s, 'whole', 0, 'text', 'all-MiniLM-L6-v2', %s::vector)
        """,
        (posting_id, vector),
    )


def test_load_embeddings_aligns_rows_and_drops_the_unembedded(conn):
    from joblens.ml import dataset

    db.upsert_postings(conn, [make_posting(str(i)) for i in range(3)])
    ids = [
        r["id"]
        for r in conn.execute("select id from postings order by source_id").fetchall()
    ]
    # Store them out of order and skip the middle one.
    _store_vector(conn, ids[2], 0.5)
    _store_vector(conn, ids[0], 0.25)
    conn.commit()

    frame = dataset.load_postings(conn=conn)
    subset, vectors = dataset.load_embeddings(frame, conn)
    assert len(subset) == 2 and vectors.shape == (2, 384)
    assert ids[1] not in set(subset["id"])
    for row, posting_id in enumerate(subset["id"]):
        expected = 0.25 if posting_id == ids[0] else 0.5
        assert vectors[row, 0] == pytest.approx(expected)
