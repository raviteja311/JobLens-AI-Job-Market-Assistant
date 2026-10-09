"""Reading a city out of a query, and the city vote in hybrid search."""

import pytest

from joblens.search import places, retrieval
from joblens.search.chunking import chunk_sections, chunk_whole
from joblens.search.retrieval import SearchHit


@pytest.mark.parametrize(
    "query, location, expected",
    [
        ("data engineer jobs in Pune", "Pune, Maharashtra, India", True),
        ("ML engineer Bengaluru", "Bangalore, India", True),
        ("ML engineer Bangalore", "Bengaluru, Karnataka", True),
        ("analytics manager Gurgaon", "Gurugram, Haryana", True),
        ("data scientist Chennai", "Hyderabad, India", False),
        ("data engineer Pune", "Punekar Road, Mumbai", False),
    ],
)
def test_query_city_matches_every_spelling(query, location, expected):
    city = places.query_city(query)
    assert city is not None
    assert places.in_city(location, city) is expected


def test_a_query_without_a_city_has_none():
    assert places.query_city("remote LLM engineer India") is None
    assert places.query_city("") is None


def hit(posting_id, location):
    return SearchHit(
        posting_id=posting_id,
        title=f"Job {posting_id}",
        company="Acme",
        url="",
        location=location,
        is_remote=False,
        score=0.0,
    )


@pytest.fixture
def arms(monkeypatch):
    """Both arms rank 1..12 the same way; only posting 9 is in Pune."""
    ranked = [hit(i, "Pune, India" if i == 9 else "Hyderabad") for i in range(1, 13)]
    monkeypatch.setattr(
        retrieval, "keyword_search", lambda conn, q, limit: list(ranked)[:limit]
    )
    monkeypatch.setattr(
        retrieval, "vector_search", lambda conn, q, **kw: list(ranked)[: kw["limit"]]
    )


def ids(hits):
    return [h.posting_id for h in hits]


def test_an_in_city_posting_inside_the_window_is_lifted(arms, monkeypatch):
    monkeypatch.setattr(retrieval, "CITY_BOOST_WINDOW", 3)
    boosted = retrieval.hybrid_search(None, "data engineer Pune", limit=4)
    # Ninth in both arms, inside a window of 3 x 4 = 12, so the city vote
    # lifts it into the first page.
    assert 9 in ids(boosted)
    assert boosted[0].posting_id in (1, 9)


def test_outside_the_window_there_is_no_vote(arms, monkeypatch):
    monkeypatch.setattr(retrieval, "CITY_BOOST_WINDOW", 2)
    hits = retrieval.hybrid_search(None, "data engineer Pune", limit=4)
    # A window of 2 x 4 = 8 stops before posting 9.
    assert ids(hits) == [1, 2, 3, 4]


def test_no_city_or_boost_off_leaves_the_ranking_alone(arms):
    assert ids(retrieval.hybrid_search(None, "data engineer", limit=4)) == [1, 2, 3, 4]
    off = retrieval.hybrid_search(None, "data engineer Pune", limit=4, city_boost=False)
    assert ids(off) == [1, 2, 3, 4]


def test_chunk_headers_carry_the_location():
    (whole,) = chunk_whole(1, "Data Engineer", "Acme", "Build pipelines.", "Pune")
    assert whole.content.startswith("Data Engineer at Acme, Pune. ")
    sections = chunk_sections(1, "Data Engineer", "Acme", "One.\nTwo.", "Pune")
    assert all(c.content.startswith("Data Engineer at Acme, Pune. ") for c in sections)
    # No location, no trailing comma.
    (bare,) = chunk_whole(1, "Data Engineer", "Acme", "Build pipelines.")
    assert bare.content.startswith("Data Engineer at Acme. ")
