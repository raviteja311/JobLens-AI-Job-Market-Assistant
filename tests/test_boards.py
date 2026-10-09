"""The Greenhouse, Lever and Ashby sources, against saved responses.

Each fixture holds one job that belongs in the corpus and one that does not,
so every parser is checked in both directions. No test touches the network.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from joblens.sources import ashby, boards, greenhouse, lever
from joblens.sources.companies import Company

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def with_slug(job: dict, slug: str) -> dict:
    # What fetch_boards stores in bronze: the job plus the board it came from.
    return {**job, boards.SLUG_KEY: slug}


# Greenhouse ---------------------------------------------------------------


def test_greenhouse_parses_an_indian_data_role():
    job = load("greenhouse_jobs.json")["jobs"][1]
    posting = greenhouse.to_posting(with_slug(job, "sigmoid"))
    assert posting.source_id == "sigmoid:5083674002"
    assert posting.title == "Associate Lead Data Scientist"
    assert posting.company == "Sigmoid"  # from the YAML, not the API
    assert "Bengaluru" in posting.location
    assert posting.url.startswith("https://")
    assert posting.posted_at is not None
    # The escaped HTML was unescaped and stripped, not stored as "&lt;p&gt;".
    assert "&lt;" not in posting.description and "<p>" not in posting.description
    assert len(posting.description) > 500


def test_greenhouse_drops_a_us_job():
    job = load("greenhouse_jobs.json")["jobs"][0]
    assert job["location"]["name"] == "Atlanta , GA, USA"
    assert greenhouse.to_posting(with_slug(job, "sigmoid")) is None


def test_a_board_missing_from_the_company_list_is_skipped():
    job = load("greenhouse_jobs.json")["jobs"][1]
    assert greenhouse.to_posting(with_slug(job, "no-such-board")) is None


# Lever --------------------------------------------------------------------


def test_lever_parses_an_indian_analytics_role():
    job = load("lever_postings.json")[0]
    posting = lever.to_posting(with_slug(job, "hevodata"))
    assert posting.source_id == f"hevodata:{job['id']}"
    assert posting.title == "Analytics Manager"
    assert posting.company == "Hevo Data"
    assert posting.posted_at == datetime.fromtimestamp(
        job["createdAt"] / 1000, tz=timezone.utc
    )
    # The titled lists are part of the description, not dropped.
    for section in job["lists"]:
        assert section["text"].split()[0] in posting.description


def test_lever_drops_a_us_job():
    job = load("lever_postings.json")[1]
    assert lever.to_posting(with_slug(job, "hevodata")) is None


def test_lever_reads_every_location():
    job = {"categories": {"location": "London", "allLocations": ["London", "Pune"]}}
    assert lever._location(job) == "London; Pune"


# Ashby --------------------------------------------------------------------


def test_ashby_parses_an_indian_data_role():
    job = load("ashby_job_board.json")["jobs"][0]
    posting = ashby.to_posting(with_slug(job, "sarvam"))
    assert posting.source_id == f"sarvam:{job['id']}"
    assert posting.title == "Data Scientist - Evaluations, Chanakya"
    assert posting.company == "Sarvam AI"
    assert posting.posted_at == datetime(2026, 4, 17, 11, 56, 48, 684000, timezone.utc)
    assert posting.description


def test_ashby_drops_a_role_outside_the_target_family():
    job = load("ashby_job_board.json")["jobs"][1]
    assert job["title"].strip() == "Product Designer"
    assert ashby.to_posting(with_slug(job, "sarvam")) is None


def test_ashby_skips_an_unlisted_job():
    job = {**load("ashby_job_board.json")["jobs"][0], "isListed": False}
    assert ashby.to_posting(with_slug(job, "sarvam")) is None


def test_ashby_uses_the_address_country():
    # A city the India filter does not know, rescued by the postal address.
    job = {
        **load("ashby_job_board.json")["jobs"][0],
        "location": "Remote",
        "address": {"postalAddress": {"addressCountry": "India"}},
    }
    assert ashby.to_posting(with_slug(job, "sarvam")) is not None


# The shared fetch loop ----------------------------------------------------


@pytest.fixture
def fake_boards(monkeypatch):
    """Three boards: one with a usable job and a US job, one that 404s, and
    one with two more usable jobs."""
    responses = {
        "one": {
            "jobs": [
                {"id": 1, "t": "Data Scientist", "l": "Pune"},
                {"id": 2, "t": "Data Scientist", "l": "Boston"},
            ]
        },
        "dead": None,
        "two": {
            "jobs": [
                {"id": 3, "t": "ML Engineer", "l": "Bengaluru"},
                {"id": 4, "t": "Data Analyst", "l": "Noida"},
            ]
        },
    }
    requested = []

    def fake_get_json(http, url, *, params=None, attempts=3):
        slug = url.rsplit("/", 1)[1]
        requested.append(slug)
        if responses[slug] is None:
            request = httpx.Request("GET", url)
            raise httpx.HTTPStatusError(
                "404", request=request, response=httpx.Response(404, request=request)
            )
        return responses[slug]

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(boards, "get_json", fake_get_json)
    monkeypatch.setattr(boards, "client", FakeClient)
    monkeypatch.setattr(
        boards.companies,
        "for_provider",
        lambda provider: [Company(s, provider, s, "Pune") for s in responses],
    )
    return requested


def run(limit):
    return boards.fetch_boards(
        "test",
        "https://example.test/{slug}",
        jobs_of=lambda response: response["jobs"],
        title_of=lambda job: job["t"],
        location_of=lambda job: job["l"],
        limit=limit,
        delay=0,
    )


def test_fetch_keeps_every_job_and_survives_a_dead_board(fake_boards):
    items = run(limit=100)
    # The Boston job is kept in bronze; to_posting filters it later.
    assert [item.source_id for item in items] == ["one:1", "one:2", "two:3", "two:4"]
    assert items[0].payload[boards.SLUG_KEY] == "one"
    assert fake_boards == ["one", "dead", "two"]


def test_fetch_limit_counts_usable_jobs_only(fake_boards):
    # One usable job on the first board meets a limit of 1, so the run stops
    # there; the US job on that board did not count towards it.
    items = run(limit=1)
    assert [item.source_id for item in items] == ["one:1", "one:2"]
    assert fake_boards == ["one"]
