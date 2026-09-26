"""Regression tests for the bugs found in the September 2026 audit.

Each test here reproduces a failure that was confirmed against the code
before the fix landed, so a revert shows up as a red test rather than as a
500 in production. They need neither a database nor a model.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager

import httpx
import pytest
from fastapi.testclient import TestClient

from joblens import observability
from joblens.api import main as api
from joblens.llm import client as llm
from joblens.rag import chat as chat_rag
from joblens.rag import resume as resume_rag
from joblens.sources import base as sources_base

# ---------------------------------------------------------------- fixtures


class _Cursor:
    def fetchone(self):
        return {"n": 0, "usd": 0.0, "calls": 0}

    def fetchall(self):
        return []


class _Conn:
    def execute(self, *args, **kwargs):
        return _Cursor()

    def commit(self):
        pass


@contextmanager
def _up():
    yield _Conn()


class _StubEmbedder:
    name = "stub"

    def encode(self, texts):
        return [[0.0] * 384 for _ in texts]


class _DownBackend:
    name = "ollama"

    def complete(self, prompt, max_tokens, json_mode):
        raise httpx.ConnectError("connection refused")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "get_embedder", lambda: _StubEmbedder())
    monkeypatch.setattr(api.db, "connect", _up)
    monkeypatch.setattr(llm, "_log_call", lambda **kwargs: None)
    with TestClient(api.app, raise_server_exceptions=False) as client:
        yield client


# ---------------------------------------------------------------- LLM client


def test_structured_calls_report_an_unreachable_backend_as_unavailable(monkeypatch):
    # `complete` already did this; `complete_structured` re-raised the raw
    # httpx error, so /match answered 500 where /chat answered 503.
    monkeypatch.setattr(llm, "get_backend", lambda: _DownBackend())
    monkeypatch.setattr(llm, "_log_call", lambda **kwargs: None)

    from pydantic import BaseModel

    class Anything(BaseModel):
        x: int = 0

    with pytest.raises(llm.BackendUnavailable, match="ollama backend unreachable"):
        llm.complete_structured("prompt", Anything, feature="test")


def test_sonnet_5_is_priced_at_two_and_ten_per_million():
    assert llm.PRICING["claude-sonnet-5"] == (2.00, 10.00)


# ---------------------------------------------------------------- /match


def test_match_returns_503_when_the_llm_backend_is_down(client, monkeypatch):
    monkeypatch.setattr(llm, "get_backend", lambda: _DownBackend())
    response = client.post("/match", files={"file": ("cv.txt", b"python aws " * 20)})
    assert response.status_code == 503
    assert "unreachable" in response.json()["detail"]


def test_match_rejects_a_corrupt_pdf_with_422_not_500(client, monkeypatch):
    monkeypatch.setattr(llm, "get_backend", lambda: _DownBackend())
    response = client.post(
        "/match", files={"file": ("cv.pdf", b"%PDF-1.4 this is not really a pdf")}
    )
    assert response.status_code == 422
    assert "could not read that PDF" in response.json()["detail"]


def test_pdf_to_text_turns_pypdf_errors_into_value_errors():
    with pytest.raises(ValueError, match="could not read that PDF"):
        resume_rag.pdf_to_text(b"not a pdf at all")


def test_a_slow_match_does_not_block_other_requests(client, monkeypatch):
    # /match was `async def` calling blocking LLM code, which parked the whole
    # event loop for the duration of every resume. A sync handler runs in the
    # threadpool and /metrics answers while the match is still running.
    def slow_match(conn, text, embedder=None, **kwargs):
        time.sleep(1.5)
        return resume_rag.MatchReport(profile=resume_rag.ResumeProfile())

    monkeypatch.setattr(api.resume_rag, "match_resume", slow_match)

    worker = threading.Thread(
        target=lambda: client.post("/match", files={"file": ("cv.txt", b"python")})
    )
    worker.start()
    time.sleep(0.2)
    began = time.perf_counter()
    assert client.get("/metrics").status_code == 200
    took = time.perf_counter() - began
    worker.join()
    assert took < 1.0, f"/metrics waited {took:.2f}s behind /match"


def test_rate_limiter_forgets_expired_callers(monkeypatch):
    monkeypatch.setattr(api, "_SWEEP_ABOVE", 2)
    api._hits.clear()
    long_ago = time.monotonic() - 3600
    for ip in ("10.0.0.1", "10.0.0.2", "10.0.0.3"):
        api._hits[ip].append(long_ago)

    class _Client:
        host = "10.0.0.9"

    class _Request:
        client = _Client()

    api.rate_limit(_Request())
    assert set(api._hits) == {"10.0.0.9"}
    api._hits.clear()


# ---------------------------------------------------------------- chat


def test_a_source_with_no_location_is_not_labelled_remote():
    kwargs = dict(n=1, posting_id=1, title="ML Eng", company="Acme", url="u", text="t")
    unknown = chat_rag.Source(location=None, **kwargs)
    remote = chat_rag.Source(location=None, is_remote=True, **kwargs)
    assert unknown.render().splitlines()[0] == "[1] ML Eng at Acme"
    assert remote.render().splitlines()[0] == "[1] ML Eng at Acme (remote)"


# ---------------------------------------------------------------- sources


def test_retry_after_accepts_seconds_dates_and_garbage():
    assert sources_base._retry_delay("3", 2.0) == 3.0
    assert sources_base._retry_delay(None, 2.0) == 2.0
    assert sources_base._retry_delay("not a number", 2.0) == 2.0
    assert sources_base._retry_delay("nan", 2.0) == 2.0
    assert sources_base._retry_delay("inf", 2.0) == 2.0
    # An HTTP-date in the past means "now", not a crash and not a negative sleep.
    assert sources_base._retry_delay("Wed, 21 Oct 2015 07:28:00 GMT", 2.0) == 0.0
    # And nobody waits an hour inside a cron job.
    assert sources_base._retry_delay("3600", 2.0) == sources_base.MAX_RETRY_DELAY


def test_get_json_survives_an_http_date_retry_after(monkeypatch):
    monkeypatch.setattr(sources_base.time, "sleep", lambda seconds: None)

    def handler(request):
        return httpx.Response(
            429, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="giving up"):
        sources_base.get_json(http, "https://example.test/api", attempts=2)


# ---------------------------------------------------------------- logging


def test_log_level_is_case_insensitive_and_falls_back_to_info():
    assert observability._coerce_level("debug") == logging.DEBUG
    assert observability._coerce_level("WARNING") == logging.WARNING
    assert observability._coerce_level("loud") == logging.INFO
    assert observability._coerce_level(logging.ERROR) == logging.ERROR
