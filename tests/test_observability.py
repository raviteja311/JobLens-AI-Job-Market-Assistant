"""Phase 7: logs and metrics, tested without a server or a database.

The API tests here build the real FastAPI app with the embedder stubbed out,
so the middleware, /metrics and the body-size guard run exactly as deployed.
The database collector is exercised through a fake connection: what matters
is that a scrape survives Postgres being down, and that is easiest to prove
by making it down.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from joblens import observability
from joblens.api import main as api

# ---------------------------------------------------------------- logging


def _format(record_kwargs, extra=None, fmt=None):
    formatter = fmt or observability.JsonFormatter()
    record = logging.LogRecord(
        name="joblens.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=record_kwargs.get("msg", "hello %s"),
        args=record_kwargs.get("args", ("world",)),
        exc_info=None,
    )
    for key, value in (extra or {}).items():
        setattr(record, key, value)
    return formatter.format(record)


def test_json_lines_are_json_with_stable_keys():
    line = json.loads(_format({}))
    assert line["msg"] == "hello world"
    assert line["level"] == "INFO"
    assert line["logger"] == "joblens.test"
    # ISO 8601 with a timezone, so two hosts' logs interleave correctly.
    assert datetime.fromisoformat(line["ts"]).tzinfo is not None


def test_extra_fields_become_top_level_keys():
    line = json.loads(_format({}, extra={"route": "/search", "status": 200}))
    assert line["route"] == "/search"
    assert line["status"] == 200
    # And the LogRecord's own attributes do not leak in beside them.
    assert "levelno" not in line and "args" not in line


def test_unserialisable_extras_do_not_break_the_line():
    when = datetime(2026, 9, 22, tzinfo=timezone.utc)
    line = json.loads(_format({}, extra={"when": when}))
    assert line["when"].startswith("2026-09-22")


def test_configure_logging_is_idempotent_and_switchable():
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        observability.configure_logging("INFO", "json")
        observability.configure_logging("INFO", "json")
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, observability.JsonFormatter)
        observability.configure_logging("INFO", "text")
        assert len(root.handlers) == 1
        assert not isinstance(root.handlers[0].formatter, observability.JsonFormatter)
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in before:
            root.addHandler(handler)


# ---------------------------------------------------------------- collector


@contextmanager
def _down():
    raise ConnectionError("connection refused")
    yield  # pragma: no cover


class _Cursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class _Conn:
    """Answers the collector's four queries in the order it asks them."""

    def __init__(self):
        self.finished = datetime(2026, 9, 22, 6, 4, tzinfo=timezone.utc)
        self.answers = [
            [{"n": 465}],
            [{"usd": 0.0123, "calls": 4}],
            [
                {
                    "source": "hackernews",
                    "status": "ok",
                    "started_at": self.finished,
                    "finished_at": self.finished,
                    "inserted": 12,
                },
                {
                    "source": "remoteok",
                    "status": "failed",
                    "started_at": self.finished,
                    "finished_at": self.finished,
                    "inserted": 0,
                },
            ],
            [{"source": "hackernews", "finished_at": self.finished}],
        ]

    def execute(self, sql, *params):
        return _Cursor(self.answers.pop(0))


@contextmanager
def _up():
    yield _Conn()


def _samples(collector):
    out = {}
    for family in collector.collect():
        for sample in family.samples:
            out[(sample.name, tuple(sorted(sample.labels.items())))] = sample.value
    return out


def test_a_scrape_survives_the_database_being_down():
    samples = _samples(observability.DatabaseCollector(_down))
    assert samples == {("joblens_db_up", ()): 0}


def test_the_collector_reports_ingest_health_per_source():
    samples = _samples(observability.DatabaseCollector(_up))
    assert samples[("joblens_db_up", ())] == 1
    assert samples[("joblens_postings", ())] == 465
    assert samples[("joblens_llm_cost_usd_today", ())] == pytest.approx(0.0123)
    assert samples[("joblens_llm_calls_today", ())] == 4
    assert samples[("joblens_ingest_last_run_ok", (("source", "hackernews"),))] == 1
    assert samples[("joblens_ingest_last_run_ok", (("source", "remoteok"),))] == 0
    stamp = samples[
        ("joblens_ingest_last_success_timestamp_seconds", (("source", "hackernews"),))
    ]
    assert stamp == _Conn().finished.timestamp()
    # remoteok never succeeded, so it has no timestamp rather than a zero
    # that an alert would read as "succeeded in 1970".
    assert (
        "joblens_ingest_last_success_timestamp_seconds",
        (("source", "remoteok"),),
    ) not in samples


def test_llm_calls_land_on_the_counters():
    before = observability.LLM_COST.labels("test", "stub", "m")._value.get()
    observability.record_llm_call(
        feature="test",
        backend="stub",
        model="m",
        ok=True,
        cost_usd=0.002,
        latency_ms=1500,
    )
    after = observability.LLM_COST.labels("test", "stub", "m")._value.get()
    assert after - before == pytest.approx(0.002)


# ---------------------------------------------------------------- the app


class _StubEmbedder:
    name = "stub"

    def encode(self, texts):
        return [[0.0] * 384 for _ in texts]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "get_embedder", lambda: _StubEmbedder())
    monkeypatch.setattr(api, "warm_reranker", lambda: None)
    monkeypatch.setattr(api.db, "connect", _down)
    with TestClient(api.app, raise_server_exceptions=False) as client:
        yield client


def test_metrics_endpoint_serves_prometheus_text(client):
    client.get("/search?q=python")  # a real request first, so there is a sample
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "joblens_http_request_duration_seconds_bucket" in body
    assert 'route="/search"' in body
    # The database is down in this test; the scrape says so and still works.
    assert "joblens_db_up 0.0" in body


def test_unmatched_paths_do_not_create_a_series_each(client):
    client.get("/wp-admin/login.php")
    client.get("/another/random/path")
    body = client.get("/metrics").text
    assert 'route="unmatched"' in body
    assert "wp-admin" not in body


def test_oversized_bodies_are_refused_before_they_are_read(client):
    response = client.post(
        "/match",
        content=b"x",
        headers={"content-length": str(50 * 1024 * 1024)},
    )
    assert response.status_code == 413
    assert "under" in response.json()["detail"]


def test_request_logs_carry_route_status_and_duration(client, caplog):
    with caplog.at_level(logging.INFO, logger="joblens.api.main"):
        client.get("/search?q=python&mode=vector")
    records = [r for r in caplog.records if r.getMessage() == "request"]
    assert records, "no request line was logged"
    record = records[-1]
    assert record.route == "/search"
    assert record.status == 500  # the database is down in this fixture
    assert record.duration_ms >= 0
    assert record.path == "/search"


def test_health_reports_which_llm_is_configured_and_the_version(client):
    # Health needs the database; with it down the endpoint fails loudly
    # rather than returning "ok" for a service that cannot serve.
    assert client.get("/health").status_code == 500


def test_chat_returns_503_when_the_llm_backend_is_down(client, monkeypatch):
    from joblens.llm.client import BackendUnavailable

    def down(*args, **kwargs):
        raise BackendUnavailable("ollama backend unreachable: connection refused")

    monkeypatch.setattr(api.chat_rag, "ask", down)
    # The fixture's database is down too; /chat opens a connection first, so
    # swap in one that yields without connecting.
    monkeypatch.setattr(api.db, "connect", _up)
    response = client.post("/chat", json={"question": "who is hiring?"})
    assert response.status_code == 503
    assert "unreachable" in response.json()["detail"]
