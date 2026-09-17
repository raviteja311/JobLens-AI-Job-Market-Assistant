"""End to end pipeline tests.
The network is stubbed out with the same fixtures the parser tests use, so
this exercises fetch -> bronze -> transform -> silver -> run log for real
against Postgres without depending on a job board being up.
"""

import json
from pathlib import Path

import pytest

from joblens import db, pipeline
from joblens.sources import remoteok

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
        conn.execute("truncate postings, raw_postings, ingestion_runs")
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
    pipeline.ingest_source("remoteok", limit=50)
    second = pipeline.ingest_source("remoteok", limit=50)
    assert second.inserted == 0
    assert second.updated == 2
    with db.connect() as conn:
        silver = conn.execute("select count(*) as n from postings").fetchone()["n"]
        bronze = conn.execute("select count(*) as n from raw_postings").fetchone()["n"]
    assert silver == 2
    # Bronze keeps both fetches. That is the point of bronze.
    assert bronze == 6


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
