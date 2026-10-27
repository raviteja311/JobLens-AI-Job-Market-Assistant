"""End to end pipeline tests.
The network is stubbed out with the same fixtures the parser tests use, so
this exercises fetch -> bronze -> transform -> silver -> run log for real
against Postgres without depending on a job board being up.
"""

import json
from pathlib import Path

import pytest

from joblens import db, pipeline
from joblens.models import Posting, RawItem
from joblens.sources import boards, greenhouse, lever, remoteok

FIXTURES = Path(__file__).parent / "fixtures"
pytestmark = pytest.mark.db


def _database_available() -> bool:
    try:
        with db.connect() as conn:
            conn.execute("select 1")
        return True
    except Exception:
        return False


if not _database_available():
    pytest.skip("no Postgres reachable at DATABASE_URL", allow_module_level=True)


@pytest.fixture
def clean_db():
    db.migrate()
    with db.connect() as conn:
        conn.execute("truncate postings, raw_postings, ingestion_runs cascade")
        conn.commit()
    yield


@pytest.fixture
def offline_remoteok(monkeypatch):
    payload = json.loads((FIXTURES / "remoteok.json").read_text())
    monkeypatch.setattr(remoteok, "get_json", lambda *a, **k: payload)


def test_full_run_lands_clean_postings(clean_db, offline_remoteok):
    result = pipeline.ingest_source("remoteok", limit=50)
    assert not result.failed
    assert result.fetched == 3  # everything with an id goes to bronze
    assert result.inserted == 2  # one entry has no title, so it never reaches silver
    assert result.skipped == 1
    with db.connect() as conn:
        row = conn.execute(
            "select title, company, salary_min, is_remote from postings"
            " order by source_id"
        ).fetchone()
    assert row["title"] == "Machine Learning Engineer"
    assert row["salary_min"] == 90000
    assert row["is_remote"] is True


def test_second_run_updates_instead_of_duplicating(clean_db, offline_remoteok):
    first = pipeline.ingest_source("remoteok", limit=50)
    with db.connect() as conn:
        seen_before = conn.execute(
            "select max(last_seen_at) as t from postings"
        ).fetchone()["t"]
    second = pipeline.ingest_source("remoteok", limit=50)
    assert first.stored == 3
    assert second.fetched == 3
    assert second.stored == 0  # nothing changed upstream
    assert second.inserted == 0
    assert second.updated == 2
    with db.connect() as conn:
        silver = conn.execute("select count(*) as n from postings").fetchone()["n"]
        bronze = conn.execute("select count(*) as n from raw_postings").fetchone()["n"]
        seen_after = conn.execute(
            "select min(last_seen_at) as t from postings"
        ).fetchone()["t"]
    assert silver == 2
    # An identical copy is not stored twice; bronze records changes only.
    assert bronze == 3
    # But every job seen again is still marked as seen.
    assert seen_after > seen_before


def test_a_changed_job_is_stored_again(clean_db, monkeypatch):
    payload = json.loads((FIXTURES / "remoteok.json").read_text())
    monkeypatch.setattr(remoteok, "get_json", lambda *a, **k: payload)
    pipeline.ingest_source("remoteok", limit=50)
    edited = [dict(entry) for entry in payload]
    edited[1]["position"] = "Senior Machine Learning Engineer"
    monkeypatch.setattr(remoteok, "get_json", lambda *a, **k: edited)
    second = pipeline.ingest_source("remoteok", limit=50)
    assert second.stored == 1
    with db.connect() as conn:
        bronze = conn.execute("select count(*) as n from raw_postings").fetchone()["n"]
    assert bronze == 4


def test_run_is_recorded_in_the_log(clean_db, offline_remoteok):
    pipeline.ingest_source("remoteok", limit=50)
    with db.connect() as conn:
        row = conn.execute(
            "select source, status, fetched, inserted from ingestion_runs"
            " order by started_at desc"
        ).fetchone()
    assert row["source"] == "remoteok"
    assert row["status"] == "ok"
    assert row["fetched"] == 3


def test_failure_is_recorded_and_does_not_raise(clean_db, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("upstream is down")

    monkeypatch.setattr(remoteok, "get_json", boom)
    result = pipeline.ingest_source("remoteok", limit=10)
    assert result.failed
    with db.connect() as conn:
        row = conn.execute("select status, error from ingestion_runs").fetchone()
    assert row["status"] == "failed"
    assert "upstream is down" in row["error"]


def test_transform_reparses_bronze_without_the_network(clean_db, offline_remoteok):
    pipeline.ingest_source("remoteok", limit=50)
    # Simulate a bad parse landing in silver, then fixing the parser and
    # re-running the transform. This is the workflow bronze exists for.
    with db.connect() as conn:
        conn.execute("update postings set salary_min = null, title = 'WRONG'")
        conn.commit()
    result = pipeline.transform("remoteok")
    assert result.updated == 2
    with db.connect() as conn:
        row = conn.execute(
            "select title, salary_min from postings where source_id = '1099821'"
        ).fetchone()
    assert row["title"] == "Machine Learning Engineer"
    assert row["salary_min"] == 90000


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="unknown source"):
        pipeline.ingest_source("linkedin")


def test_a_database_error_mid_run_is_recorded_and_does_not_raise(
    clean_db, offline_remoteok, monkeypatch
):
    # A failing statement leaves psycopg in an aborted transaction. Without a
    # rollback the run-log update is refused too, the row stays 'running'
    # forever, and the exception escapes to abort the remaining sources.
    def broken_upsert(conn, postings):
        conn.execute("select * from this_table_does_not_exist")
        return (0, 0)

    monkeypatch.setattr(pipeline.db, "upsert_postings", broken_upsert)
    result = pipeline.ingest_source("remoteok", limit=10)
    assert result.failed
    assert "this_table_does_not_exist" in result.error
    with db.connect() as conn:
        row = conn.execute("select status, error from ingestion_runs").fetchone()
    assert row["status"] == "failed"
    assert "this_table_does_not_exist" in row["error"]


# Pruning. A tightened filter is simulated by monkeypatching the source's
# is_target_role, never by editing fixtures: the bronze payload stays the
# same and only the rule that judges it changes.

GREENHOUSE_SLUG = "razorpaysoftwareprivatelimited"
LEVER_SLUG = "hevodata"
KEPT_TITLE = "Senior Data Engineer"
REJECTED_TITLE = "Associate Lead Data Scientist"


def _greenhouse_jobs() -> list[dict]:
    india_job = json.loads(
        (FIXTURES / "greenhouse_jobs.json").read_text(encoding="utf-8")
    )["jobs"][1]
    assert india_job["title"] == REJECTED_TITLE
    kept = {**india_job, "id": 1, "title": KEPT_TITLE}
    return [india_job, kept]


def _lever_job() -> dict:
    return json.loads((FIXTURES / "lever_postings.json").read_text(encoding="utf-8"))[0]


def _load_bronze(source: str, slug: str, jobs: list[dict]) -> None:
    with db.connect() as conn:
        run_id = db.start_run(conn, source)
        items = [
            RawItem(
                source=source,
                source_id=f"{slug}:{job['id']}",
                payload={**job, boards.SLUG_KEY: slug},
            )
            for job in jobs
        ]
        db.insert_raw(conn, items, run_id)
        conn.commit()


def _reject(monkeypatch, module, *titles: str) -> None:
    original = module.is_target_role
    monkeypatch.setattr(
        module,
        "is_target_role",
        lambda title: title not in titles and original(title),
    )


def _postings() -> dict[str, dict]:
    with db.connect() as conn:
        rows = conn.execute(
            "select id, source, source_id, title from postings"
        ).fetchall()
    return {f"{r['source']}:{r['source_id']}": r for r in rows}


def _add_chunk(posting_id: int) -> None:
    vector = "[" + ",".join(["0"] * 384) + "]"
    with db.connect() as conn:
        conn.execute(
            "insert into posting_chunks"
            " (posting_id, strategy, chunk_index, content, model, embedding)"
            " values (%s, 'whole', 0, 'text', 'test-model', %s::vector)",
            (posting_id, vector),
        )
        conn.commit()


def _chunk_count() -> int:
    with db.connect() as conn:
        return conn.execute("select count(*) as n from posting_chunks").fetchone()["n"]


@pytest.fixture
def greenhouse_in_silver(clean_db):
    _load_bronze("greenhouse", GREENHOUSE_SLUG, _greenhouse_jobs())
    pipeline.transform("greenhouse")
    postings = _postings()
    assert len(postings) == 2, postings
    return postings


REJECTED_KEY = f"greenhouse:{GREENHOUSE_SLUG}:5083674002"
KEPT_KEY = f"greenhouse:{GREENHOUSE_SLUG}:1"


def test_prune_deletes_a_posting_the_filter_now_rejects(
    greenhouse_in_silver, monkeypatch
):
    _add_chunk(greenhouse_in_silver[REJECTED_KEY]["id"])
    _reject(monkeypatch, greenhouse, REJECTED_TITLE)
    result = pipeline.transform(prune=True)
    assert result.pruned == 1
    assert set(_postings()) == {KEPT_KEY}
    assert _chunk_count() == 0  # the chunk went with its posting


def test_without_prune_a_rejected_posting_is_kept(greenhouse_in_silver, monkeypatch):
    _reject(monkeypatch, greenhouse, REJECTED_TITLE)
    result = pipeline.transform()
    assert result.pruned == 0
    assert set(_postings()) == {REJECTED_KEY, KEPT_KEY}


def test_dry_run_reports_and_writes_nothing(greenhouse_in_silver, monkeypatch):
    _add_chunk(greenhouse_in_silver[REJECTED_KEY]["id"])
    # A stale title the transform would repair, to prove upserts roll back too.
    with db.connect() as conn:
        conn.execute("update postings set title = 'STALE'")
        conn.commit()
    _reject(monkeypatch, greenhouse, REJECTED_TITLE)
    result = pipeline.transform(prune=True, dry_run=True)
    assert result.pruned == 1
    assert [
        (p["source"], p["source_id"], p["title"]) for p in result.pruned_postings
    ] == [("greenhouse", f"{GREENHOUSE_SLUG}:5083674002", "STALE")]
    after = _postings()
    assert set(after) == {REJECTED_KEY, KEPT_KEY}
    assert {p["title"] for p in after.values()} == {"STALE"}
    assert _chunk_count() == 1


def test_prune_with_a_source_leaves_other_sources_alone(
    greenhouse_in_silver, monkeypatch
):
    _load_bronze("lever", LEVER_SLUG, [_lever_job()])
    pipeline.transform("lever")
    lever_key = f"lever:{LEVER_SLUG}:{_lever_job()['id']}"
    assert lever_key in _postings()
    # Both filters now reject everything, but only greenhouse is pruned.
    monkeypatch.setattr(greenhouse, "is_target_role", lambda title: False)
    monkeypatch.setattr(lever, "is_target_role", lambda title: False)
    result = pipeline.transform("greenhouse", prune=True)
    assert result.pruned == 2
    assert set(_postings()) == {lever_key}


def test_a_posting_with_no_bronze_row_is_never_pruned(
    greenhouse_in_silver, monkeypatch
):
    orphan = Posting.build(
        source="greenhouse",
        source_id=f"{GREENHOUSE_SLUG}:999",
        title="Data Scientist",
        company="Razorpay",
        url="https://example.com/999",
        location="Bengaluru, India",
        description_html="<p>Build models.</p>",
    )
    with db.connect() as conn:
        db.upsert_postings(conn, [orphan])
    monkeypatch.setattr(greenhouse, "is_target_role", lambda title: False)
    result = pipeline.transform(prune=True)
    assert result.pruned == 2
    assert result.without_bronze == 1
    assert set(_postings()) == {f"greenhouse:{GREENHOUSE_SLUG}:999"}
