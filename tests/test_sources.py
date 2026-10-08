"""Source parser tests.
Every one of these runs against a saved fixture, so the suite never touches
the network and never fails because a job board is down.
To refresh a fixture:
    curl https://remoteok.com/api > tests/fixtures/remoteok.json
"""

import json
from pathlib import Path

import pytest

from joblens.sources import adzuna, hackernews, remoteok

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def remoteok_jobs():
    return [entry for entry in load("remoteok.json") if entry.get("id")]


def test_remoteok_parses_a_full_posting(remoteok_jobs):
    posting = remoteok.to_posting(remoteok_jobs[0])
    assert posting.title == "Machine Learning Engineer"
    assert posting.company == "Northwind Analytics"
    assert posting.is_remote is True
    assert posting.salary_min == 90000
    assert posting.salary_max == 120000
    assert posting.salary_currency == "USD"
    assert "PyTorch" in posting.description
    assert "<p>" not in posting.description
    assert posting.posted_at is not None


def test_remoteok_handles_zero_salary(remoteok_jobs):
    # The board uses 0 for "not stated", which must not become a 0 salary.
    posting = remoteok.to_posting(remoteok_jobs[1])
    assert posting.salary_min is None
    assert posting.salary_raw is None


def test_remoteok_skips_entries_without_a_title(remoteok_jobs):
    assert remoteok.to_posting(remoteok_jobs[2]) is None


def test_remoteok_fetch_skips_the_legal_notice(monkeypatch):
    payload = load("remoteok.json")
    monkeypatch.setattr(remoteok, "get_json", lambda *a, **k: payload)
    items = remoteok.fetch(limit=10)
    assert len(items) == 3
    assert all(item.source == "remoteok" for item in items)


@pytest.fixture
def hn_comments():
    return load("hackernews_thread.json")["children"]


def test_hn_parses_the_pipe_convention(hn_comments):
    posting = hackernews.to_posting(hn_comments[0])
    assert posting.company == "Acme Robotics"
    assert posting.title == "Senior ML Engineer"
    assert posting.location == "London, UK"
    assert posting.is_remote is True
    assert posting.salary_currency == "GBP"
    assert posting.salary_min == 90000


def test_hn_handles_a_non_remote_posting(hn_comments):
    posting = hackernews.to_posting(hn_comments[1])
    assert posting.company == "Tessellate"
    assert posting.is_remote is False
    assert posting.salary_currency == "EUR"


def test_hn_ignores_comments_that_are_not_job_posts(hn_comments):
    assert hackernews.to_posting(hn_comments[2]) is None


def test_hn_ignores_deleted_comments(hn_comments):
    assert hackernews.to_posting(hn_comments[3]) is None


def test_adzuna_parses_a_posting():
    entry = load("adzuna_search.json")["results"][0]
    posting = adzuna.to_posting(entry)
    assert posting.company == "Halcyon Data Ltd"
    assert posting.salary_min == 55000
    assert posting.salary_max == 70000
    assert posting.salary_period == "year"


def test_adzuna_discards_predicted_salaries():
    # Adzuna's own estimate, not something the employer published. Training a
    # salary model on another model's output teaches you nothing.
    entry = load("adzuna_search.json")["results"][1]
    posting = adzuna.to_posting(entry)
    assert posting.salary_min is None


def test_adzuna_is_skipped_without_credentials(monkeypatch):
    monkeypatch.delenv("ADZUNA_APP_ID", raising=False)
    monkeypatch.delenv("ADZUNA_APP_KEY", raising=False)
    adzuna.get_settings.cache_clear()
    assert adzuna.fetch(limit=10) == []


@pytest.fixture
def adzuna_pages(monkeypatch):
    """Fake Adzuna: two queries, two pages each, where "b" repeats job 2."""
    pages = {
        ("a", 1): [{"id": 1}, {"id": 2}],
        ("a", 2): [{"id": 3}],
        ("b", 1): [{"id": 2}, {"id": 4}],
        ("b", 2): [{"id": 5}],
    }
    calls = []

    def fake_get_json(http, url, *, params=None, attempts=3):
        page = int(url.rsplit("/", 1)[1])
        calls.append((params["what"], page))
        assert "/jobs/in/" in url
        return {"results": pages.get((params["what"], page), [])}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setenv("ADZUNA_APP_ID", "test-id")
    monkeypatch.setenv("ADZUNA_APP_KEY", "test-key")
    monkeypatch.setenv("ADZUNA_COUNTRY", "in")
    monkeypatch.setenv("ADZUNA_QUERIES", '["a", "b"]')
    monkeypatch.setattr(adzuna, "get_json", fake_get_json)
    monkeypatch.setattr(adzuna, "client", FakeClient)
    monkeypatch.setattr(adzuna, "RESULTS_PER_PAGE", 2)
    adzuna.get_settings.cache_clear()
    yield calls
    adzuna.get_settings.cache_clear()


def test_adzuna_runs_every_query_and_keeps_a_job_once(adzuna_pages):
    items = adzuna.fetch(limit=100)
    assert [item.source_id for item in items] == ["1", "2", "3", "4", "5"]
    assert {query for query, _ in adzuna_pages} == {"a", "b"}


def test_adzuna_splits_the_limit_across_queries(adzuna_pages):
    # A limit of 4 is 2 per query, so "a" must not take all four slots.
    items = adzuna.fetch(limit=4)
    assert [item.source_id for item in items] == ["1", "2", "4", "5"]
