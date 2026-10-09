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
            "location": "India - Bengaluru",
            "cities": ["Bengaluru"],
            "is_remote": True,
            "score": 0.03,
            "ranks": {"vector": 1, "keyword": 4},
            "snippet": "Acme | ML | Bengaluru\nWe <b>build</b> models. "
            "[Apply here: https://jobs.example.com/1?ref=hn] www.example.com/x",
        },
        {
            "posting_id": 2,
            "title": "Data Engineer",
            "company": "Beta",
            "url": "https://example.com/2",
            "location": "Pune, IN",
            "cities": ["Pune"],
            "is_remote": False,
            "score": 0.02,
            "ranks": {"keyword": 1},
            "snippet": "",
        },
    ],
}
TREND_SUMMARY = {
    "window_days": 90,
    "postings": 3,
    "sources": {"hackernews": 3},
    "remote_share": 0.5,
    "top_skills": [{"skill": "python", "postings": 3, "share": 1.0}],
    "top_regions": [
        {"region": "Bengaluru", "postings": 2, "remote_share": 0.5},
        {"region": "City not stated", "postings": 1, "remote_share": 1.0},
    ],
}


def _get(url, params=None, timeout=None, health=HEALTH):
    if url.endswith("/health"):
        body = health
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


def _search(act=None):
    """Render the search tab, optionally after changing its widgets."""
    at = AppTest.from_file(APP, default_timeout=30)
    with mock.patch("httpx.get", side_effect=_get):
        at.run()
        if act:
            act(at)
            at.run()
    assert not at.exception, at.exception
    return at


def _toggle(at, label):
    return next(t for t in at.toggle if t.label == label)


def test_search_card_escapes_the_title_and_names_the_city():
    at = _search()
    md = _markdown(at)
    assert any(r"ML \[Senior\] \*Engineer\*" in m and r"Acme\_Co" in m for m in md)
    # The API's canonical city, not the board's "India - Bengaluru".
    assert any(":material/location_on: Bengaluru]" in m for m in md)
    # The ranking internals stay out of the way until asked for.
    assert not any("found by" in m for m in md)
    # The first snippet line repeats company and title; <b> tags and URLs go.
    assert any(c.value == "We build models." for c in at.caption)


def test_the_ranking_is_explained_on_request():
    at = _search(lambda at: _toggle(at, "Show why each result matched").set_value(True))
    assert any("found by keyword #4 · vector #1" in m for m in _markdown(at))


def test_search_filters_by_city_and_remote():
    at = _search(lambda at: at.selectbox[0].set_value("Pune"))
    assert at.selectbox[0].options == ["Any city", "Bengaluru", "Pune"]
    md = _markdown(at)
    assert any("Data Engineer" in m for m in md)
    assert not any("Acme" in m for m in md)
    assert any(c.value.startswith("1 of 2 results") for c in at.caption)

    at = _search(lambda at: _toggle(at, "Remote only").set_value(True))
    md = _markdown(at)
    assert any("Acme" in m for m in md)
    assert not any("Data Engineer" in m for m in md)


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


def test_only_cited_postings_are_listed_as_sources():
    def posting(n):
        return {
            "n": n,
            "posting_id": n,
            "title": f"Role {n}",
            "company": "Acme",
            "url": f"https://example.com/{n}",
        }

    reply = {
        "question": "q",
        "answer": "Acme is hiring [1] and so is Beta [3].",
        "grounded": True,
        "took_ms": 900,
        "cost_usd": 0.0001,
        # What the API sends: the cited postings, and everything retrieved.
        "citations": [posting(1), posting(3)],
        "retrieved": [posting(1), posting(2), posting(3)],
    }
    at, _ = _run(CHAT, _post(200, reply), _ask)
    md = _markdown(at)
    sources = md.index("**Sources**")
    assert md[sources + 1].startswith("1. [Role 1](https://example.com/1)")
    assert md[sources + 2].startswith("3. [Role 3](https://example.com/3)")
    # The uncited one is still reachable, after the sources and under a label
    # that says so. AppTest does not list expanders that carry an icon in
    # `at.expander`, so the label is checked in the element tree.
    assert md[sources + 3].startswith("2. [Role 2](https://example.com/2)")
    assert "Also retrieved, not cited (1)" in repr(at._tree)


@pytest.mark.parametrize(
    "status, detail, expected",
    [
        (429, "slow down", "Rate limited"),
        (503, "ollama backend unreachable: [WinError 10061]", "Start the Ollama app"),
        (
            503,
            'ollama backend returned HTTP 500: {"error":"llama-server process has '
            'terminated: exit status 0xc0000409"}',
            "The model crashed while answering",
        ),
        (503, "anthropic backend failing: 529", "anthropic backend failing"),
    ],
)
def test_rate_limit_and_backend_down_are_warnings(status, detail, expected):
    at, _ = _run(CHAT, _post(status, {"detail": detail}), _ask)
    assert any(expected in w.value for w in at.warning)


def test_trends_reports_cities_honestly():
    at, _ = _run(TRENDS)
    assert [m.label for m in at.metric] == ["Postings", "Remote"]
    regions = at.dataframe[0].value
    assert regions["region"].tolist() == ["Bengaluru", "City not stated"]
    assert regions["remote_share"].tolist() == [50.0, 100.0]
    assert any("counts in each" in c.value for c in at.caption)


@pytest.mark.parametrize("tab, feature", [(CHAT, "Chat"), (MATCH, "Resume match")])
def test_without_an_llm_the_llm_tabs_say_so(tab, feature):
    off = {"postings": 697, "llm_backend": "none", "llm_enabled": False}
    at = AppTest.from_file(APP, default_timeout=30)
    with mock.patch(
        "httpx.get", side_effect=lambda url, **kw: _get(url, health=off, **kw)
    ):
        at.run()
        at.session_state["view"] = tab
        at.run()
    assert not at.exception, at.exception
    assert any(f"{feature} needs a language model" in i.value for i in at.info)
    assert not at.button


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
