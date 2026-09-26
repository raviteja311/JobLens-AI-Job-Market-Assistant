"""The Streamlit UI, rendered headlessly with the API stubbed out.

The page talks to FastAPI over HTTP, so these tests replace httpx rather
than the library: what is checked is how each API response is shown,
including the refusals and the rate-limit and backend-down replies, which
are the cases a demo never exercises.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import httpx
import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
CHAT = ":material/chat: Chat"
MATCH = ":material/description: Resume match"
TRENDS = ":material/bar_chart: Trends"

HEALTH = {"postings": 466, "llm_backend": "ollama", "llm_enabled": True}
SEARCH = {
    "took_ms": 12,
    "results": [
        {
            "posting_id": 1,
            "title": "ML [Senior] *Engineer*",
            "company": "Acme_Co",
            "url": "https://example.com/1",
            "location": "Berlin",
            "is_remote": True,
            "score": 0.03,
            "ranks": {"vector": 1, "keyword": 4},
            "snippet": "Acme | ML | Berlin\nWe <b>build</b> models. "
            "[Apply here: https://jobs.example.com/1?ref=hn] www.example.com/x",
        }
    ],
}
TREND_SUMMARY = {
    "window_days": 90,
    "postings": 3,
    "sources": {"hackernews": 3},
    "remote_share": 0.5,
    "with_salary": 1,
    "salary_coverage": 0.3,
    "median_salary_usd": 150000,
    "top_skills": [{"skill": "python", "postings": 3, "share": 1.0}],
    "top_regions": [{"region": "unknown", "postings": 2, "remote_share": 0.5}],
}


def _get(url, params=None, timeout=None):
    if url.endswith("/health"):
        body = HEALTH
    elif url.endswith("/trends"):
        body = TREND_SUMMARY
    else:
        body = SEARCH
    return httpx.Response(200, json=body, request=httpx.Request("GET", url))


def _post(status: int, body: dict) -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request("POST", "http://x"))


def _run(tab: str | None = None, post=None, act=None):
    """Render the page, optionally on another tab, optionally after an action.

    AppTest does not remember the selected tab between runs the way a
    browser does, so the tab is set again before every run.
    """
    at = AppTest.from_file(APP, default_timeout=30)
    with (
        mock.patch("httpx.get", side_effect=_get),
        mock.patch("httpx.post", return_value=post) as posted,
    ):
        at.run()
        if tab:
            at.session_state["view"] = tab
            at.run()
        if act:
            act(at)
            at.session_state["view"] = tab
            at.run()
    assert not at.exception, at.exception
    return at, posted


def _markdown(at) -> list[str]:
    return [m.value for m in at.markdown]


def test_search_card_escapes_the_title_and_explains_the_ranking():
    at, _ = _run()
    md = _markdown(at)
    assert any(r"ML \[Senior\] \*Engineer\*" in m and r"Acme\_Co" in m for m in md)
    assert any("found by keyword #4 · vector #1" in m for m in md)
    # The first snippet line repeats company and title; <b> tags and URLs go.
    assert any(c.value == "We build models." for c in at.caption)


def test_an_unreachable_api_says_how_to_start_it():
    at = AppTest.from_file(APP, default_timeout=30)
    with mock.patch("httpx.get", side_effect=httpx.ConnectError("refused")):
        at.run()
    assert "python -m joblens serve" in at.error[0].value


def _ask(at):
    at.button[0].click()


def test_a_refusal_is_shown_as_one():
    reply = {
        "question": "q",
        "answer": "The postings do not say.",
        "citations": [],
        "grounded": False,
        "took_ms": 1500,
        "cost_usd": 0,
    }
    at, _ = _run(CHAT, _post(200, reply), _ask)
    assert "The postings do not say." in _markdown(at)
    assert any("no sources cited" in c.value for c in at.caption)


def test_citations_are_listed_as_links():
    reply = {
        "question": "q",
        "answer": "Acme [1].",
        "grounded": True,
        "took_ms": 900,
        "cost_usd": 0.0001,
        "citations": [
            {
                "n": 1,
                "posting_id": 1,
                "title": "Rust Dev",
                "company": "Acme",
                "url": "https://example.com/1",
            }
        ],
    }
    at, _ = _run(CHAT, _post(200, reply), _ask)
    assert any(
        m.startswith("1. [Rust Dev](https://example.com/1)") for m in _markdown(at)
    )


@pytest.mark.parametrize(
    "status, expected",
    [(429, "Rate limited"), (503, "ollama backend unreachable")],
)
def test_rate_limit_and_backend_down_are_warnings(status, expected):
    at, _ = _run(CHAT, _post(status, {"detail": "ollama backend unreachable"}), _ask)
    assert any(expected in w.value for w in at.warning)


def test_trends_reports_regions_honestly():
    at, _ = _run(TRENDS)
    assert [m.label for m in at.metric] == [
        "Postings",
        "Remote",
        "State a salary",
        "Median salary",
    ]
    regions = at.dataframe[0].value
    assert regions["region"].tolist() == ["not stated"]
    assert regions["remote_share"].tolist() == [50.0]


def test_a_match_shows_the_fit_and_the_gaps():
    reply = {
        "resume_skills": ["python", "pytorch"],
        "took_ms": 61000,
        "cost_usd": 0,
        "matches": [
            {
                "posting_id": 1,
                "title": "ML Engineer",
                "company": "Acme",
                "url": "https://example.com/1",
                "fit_score": 82,
                "reasoning": "Strong PyTorch overlap.",
                "matched_skills": ["pytorch"],
                "missing_skills": ["kubernetes"],
            }
        ],
    }

    def upload_and_match(at):
        at.file_uploader[0].upload("cv.txt", b"python pytorch", "text/plain")
        at.session_state["view"] = MATCH
        at.run()
        [button] = [b for b in at.button if b.label == "Match my resume"]
        button.click()

    at, posted = _run(MATCH, _post(200, reply), upload_and_match)
    md = _markdown(at)
    assert posted.call_count == 1
    assert any(":blue-badge[python]" in m and ":blue-badge[pytorch]" in m for m in md)
    assert any(":green-badge[pytorch]" in m for m in md)
    assert any(":orange-badge[kubernetes]" in m for m in md)
    assert [(m.label, m.value) for m in at.metric] == [("Fit", "82/100")]
