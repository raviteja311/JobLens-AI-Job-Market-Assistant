"""The JobLens UI. Streamlit first, because a demo that exists beats a
polished one that does not.

It talks to the FastAPI backend over HTTP rather than importing joblens
directly. That costs a little latency and buys the guarantee that whatever
this page shows, the API really serves: a UI wired straight into the library
can look healthy while the deployed endpoint is broken.
"""

from __future__ import annotations

import os
import re

import httpx
import pandas as pd
import streamlit as st

API = os.environ.get("JOBLENS_API", "http://127.0.0.1:8000")
TIMEOUT = 300.0
SERVE_COMMAND = "python -m joblens serve"

st.set_page_config(page_title="JobLens", page_icon=":material/work:", layout="wide")


def api_get(path: str, **params):
    response = httpx.get(f"{API}{path}", params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


# Search and trends are read-only and the corpus changes once a day, so a
# repeated query or a tab switch should not cost another round trip. Chat
# and match are never cached: each call is an LLM answer the user asked for.
@st.cache_data(ttl=300, max_entries=200, show_spinner=False)
def cached_search(q: str, mode: str, strategy: str, rerank: bool) -> dict:
    return api_get(
        "/search", q=q, mode=mode, strategy=strategy, limit=10, rerank=rerank
    )


@st.cache_data(ttl=300, max_entries=10, show_spinner=False)
def cached_trends(days: int) -> dict:
    return api_get("/trends", days=days)


def md(text: str) -> str:
    """Escape the characters that would turn a job title into Markdown."""
    return re.sub(r"([\\`*_\[\]<>#|~$])", r"\\\1", text or "")


def badges(items: list[str], color: str) -> str:
    return " ".join(f":{color}-badge[{md(item)}]" for item in items)


def found_by(ranks: dict[str, int]) -> str:
    # Which retriever found it and at what rank. A search UI that cannot
    # explain a result is a search UI nobody trusts.
    return " · ".join(f"{name} #{rank}" for name, rank in sorted(ranks.items()))


def snippet_text(snippet: str) -> str:
    # ts_headline marks matches with <b>, the first line of a whole-posting
    # chunk repeats the company and title already shown above it, and the
    # bodies are full of application URLs.
    text = re.sub(r"</?b>", "", snippet or "").strip()
    lines = text.splitlines()
    if len(lines) > 1 and lines[0].count("|") >= 2:
        text = " ".join(lines[1:])
    # Application links are noise in a preview; the title already links out.
    # A bracketed "[Apply here: <url>]" goes whole, then any bare URL.
    text = re.sub(r"\[[^\]]*(?:https?://|www\.)[^\]]*\]?", "", text)
    text = re.sub(r"(?:https?://|www\.)\S+", "", text)
    text = " ".join(text.split())
    return text[:280] + ("..." if len(text) > 280 else "")


# Same pattern the backend uses to decide an answer is grounded
# (joblens.rag.chat._CITATION).
CITATION = re.compile(r"\[(\d+)\]")


def source_line(citation: dict) -> str:
    return (
        f"{citation['n']}. [{md(citation['title'])}]({citation['url']}) · "
        f"{md(citation['company'])}"
    )


def show_limited(response: httpx.Response, fallback: str) -> bool:
    """Render the API's own reason for a 429/413/415/422/503. True if it did."""
    if response.status_code == 429:
        st.warning("Rate limited. Try again in a minute.", icon=":material/timer:")
        return True
    if response.status_code in (413, 415, 422, 503):
        detail = response.json().get("detail", fallback)
        if response.status_code == 503 and detail.startswith(
            "ollama backend unreachable"
        ):
            # The API's reason ends in a raw socket error, which says what
            # broke but not what to do about it.
            st.warning(
                "Ollama is not running, so there is no model to answer with. "
                "Start the Ollama app (or run `ollama serve`), then ask again.",
                icon=":material/power_off:",
            )
            st.caption(detail)
        else:
            st.warning(detail, icon=":material/warning:")
        return True
    return False


st.title("JobLens")

try:
    health = api_get("/health")
except Exception as exc:  # noqa: BLE001 - the whole page depends on this
    st.error(
        f"Cannot reach the API at {API}. Start it in another terminal with "
        f"`{SERVE_COMMAND}` and refresh; it takes about 45 seconds to load "
        f"its models.\n\n{exc}",
        icon=":material/cloud_off:",
    )
    st.stop()

# Grey either way: /health knows which backend is configured, not whether it
# is up right now, and a green badge next to a dead Ollama is a small lie.
llm_badge = (
    f":gray-badge[:material/smart_toy: LLM: {health['llm_backend']}]"
    if health["llm_enabled"]
    else ":gray-badge[:material/smart_toy: LLM off]"
)
st.markdown(
    f"Real AI/ML job postings, searchable by meaning. "
    f":blue-badge[{health['postings']} postings] {llm_badge}"
)

search_tab, chat_tab, match_tab, trends_tab = st.tabs(
    [
        ":material/search: Search",
        ":material/chat: Chat",
        ":material/description: Resume match",
        ":material/bar_chart: Trends",
    ],
    on_change="rerun",
    key="view",
)


if search_tab.open:
    query = st.text_input(
        "What are you looking for?",
        "remote machine learning engineer",
        placeholder="e.g. remote LLM roles that need PyTorch",
    )
    with st.container(horizontal=True, vertical_alignment="bottom"):
        mode = st.segmented_control(
            "Retrieval",
            ["hybrid", "vector", "keyword"],
            default="hybrid",
            required=True,
            help="Hybrid adds keyword search to vector search, which catches rare "
            "terms like tool names that embeddings blur together.",
        )
        strategy = st.segmented_control(
            "Chunking", ["whole", "section"], default="whole", required=True
        )
        rerank = st.toggle(
            "Rerank",
            help="A cross-encoder reorders the results. Better top result, "
            "about 2 seconds slower.",
        )

    if query.strip():
        payload = cached_search(query.strip(), mode, strategy, rerank)
        results = payload["results"]
        st.caption(f"{len(results)} results in {payload['took_ms']:.0f} ms")
        if not results:
            st.info("No postings matched. Try fewer or broader words.")
        for result in results:
            with st.container(border=True):
                st.markdown(
                    f"**[{md(result['title'])}]({result['url']})** · "
                    f"{md(result['company'])}"
                )
                tags = []
                if result["is_remote"]:
                    tags.append(":green-badge[:material/home_work: remote]")
                if result["location"]:
                    tags.append(
                        f":gray-badge[:material/location_on: {md(result['location'])}]"
                    )
                meta = " ".join(tags)
                if result["ranks"]:
                    meta += f"  :small[found by {found_by(result['ranks'])}]"
                if meta:
                    st.markdown(meta)
                if result["snippet"]:
                    st.caption(snippet_text(result["snippet"]))


if chat_tab.open:
    st.caption(
        "Answers come only from the retrieved postings, with a numbered source "
        "for every claim. When nothing relevant is found it says so instead of "
        "guessing."
    )
    example = st.pills(
        "Examples",
        [
            "who is hiring Rust engineers?",
            "which roles involve building RAG systems or LLM agents?",
            "which companies say they offer visa sponsorship?",
        ],
        label_visibility="collapsed",
    )
    with st.form("chat", border=False):
        with st.container(horizontal=True, vertical_alignment="bottom"):
            question = st.text_input(
                "Ask about the job market",
                example or "who is hiring Rust engineers?",
            )
            asked = st.form_submit_button("Ask", type="primary", icon=":material/send:")
    if asked and question.strip():
        with st.spinner(
            "Retrieving and answering. About 50 seconds on the local model."
        ):
            response = httpx.post(
                f"{API}/chat",
                json={"question": question.strip(), "limit": 6},
                timeout=TIMEOUT,
            )
        if not show_limited(response, "LLM backend unavailable."):
            response.raise_for_status()
            body = response.json()
            with st.container(border=True):
                st.markdown(body["answer"])
                if not body["grounded"]:
                    st.caption(
                        ":orange-badge[:material/info: no sources cited] The "
                        "postings did not answer this, and the reply says so."
                    )
                # `citations` is every posting the model was given. Only the
                # ones the answer actually cites are its sources; listing all
                # six under that heading would claim support it never used.
                cited_numbers = {int(n) for n in CITATION.findall(body["answer"])}
                cited = [c for c in body["citations"] if c["n"] in cited_numbers]
                rest = [c for c in body["citations"] if c["n"] not in cited_numbers]
                if cited:
                    st.markdown("**Sources**")
                    for citation in cited:
                        st.markdown(source_line(citation))
                if rest:
                    with st.expander(
                        f"Also retrieved, not cited ({len(rest)})",
                        icon=":material/manage_search:",
                    ):
                        for citation in rest:
                            st.markdown(source_line(citation))
            st.caption(f"{body['took_ms'] / 1000:.1f} s · ${body['cost_usd']:.5f}")


if match_tab.open:
    st.caption(
        "Upload a resume to rank postings against it. The most useful part is "
        "what each posting wants that your resume does not mention."
    )
    # The API rejects anything over 2 MB (MAX_RESUME_BYTES); saying so here
    # beats letting someone upload 50 MB and get a 413.
    upload = st.file_uploader(
        "Resume (PDF or text)", type=["pdf", "txt", "md"], max_upload_size=2
    )
    if upload and st.button("Match my resume", type="primary", icon=":material/bolt:"):
        with st.spinner("Extracting skills and scoring postings. This takes a minute."):
            response = httpx.post(
                f"{API}/match",
                files={"file": (upload.name, upload.getvalue())},
                timeout=TIMEOUT,
            )
        if not show_limited(response, "Could not process that file."):
            response.raise_for_status()
            body = response.json()
            st.markdown(
                "**Skills found in your resume:** "
                + (badges(body["resume_skills"], "blue") or "none")
            )
            for match in body["matches"]:
                with st.container(border=True):
                    info, score = st.columns([5, 1], vertical_alignment="center")
                    with info:
                        st.markdown(
                            f"**[{md(match['title'])}]({match['url']})** · "
                            f"{md(match['company'])}"
                        )
                        st.write(match["reasoning"])
                    score.metric("Fit", f"{match['fit_score']}/100")
                    st.markdown(
                        "You have: "
                        + (badges(match["matched_skills"], "green") or ":small[none]")
                    )
                    st.markdown(
                        "Missing: "
                        + (badges(match["missing_skills"], "orange") or ":small[none]")
                    )
            st.caption(f"{body['took_ms'] / 1000:.1f} s · ${body['cost_usd']:.5f}")


if trends_tab.open:
    days = st.segmented_control(
        "Window",
        [30, 90, 180, 365],
        default=90,
        required=True,
        format_func=lambda d: f"{d} days",
    )
    summary = cached_trends(days)
    if not summary.get("postings"):
        st.info("No postings in this window.")
    else:
        with st.container(horizontal=True):
            st.metric("Postings", summary["postings"], border=True)
            st.metric("Remote", f"{summary['remote_share']:.0%}", border=True)
            st.metric(
                "State a salary",
                f"{summary['salary_coverage']:.0%}",
                border=True,
                help="Every figure on this tab is reported against the postings "
                "that could answer it; most postings never state pay.",
            )
            st.metric(
                "Median salary",
                (
                    f"${summary['median_salary_usd']:,}"
                    if summary["median_salary_usd"]
                    else "-"
                ),
                border=True,
                help=f"Across the {summary['with_salary']} postings that state one.",
            )

        skills_col, regions_col = st.columns([3, 2])
        with skills_col:
            st.subheader("Most requested skills")
            skills = pd.DataFrame(summary["top_skills"])
            if not skills.empty:
                st.bar_chart(
                    skills,
                    x="skill",
                    y="postings",
                    horizontal=True,
                    sort="-postings",
                    # With horizontal=True the axes swap, and their labels
                    # swap with them.
                    x_label="",
                    y_label="postings",
                )
        with regions_col:
            st.subheader("Where the jobs are")
            regions = pd.DataFrame(summary["top_regions"])
            if not regions.empty:
                regions["region"] = regions["region"].replace("unknown", "not stated")
                regions["remote_share"] = (regions["remote_share"] * 100).round()
                st.dataframe(
                    regions,
                    hide_index=True,
                    column_config={
                        "region": st.column_config.TextColumn("Region"),
                        "postings": st.column_config.NumberColumn("Postings"),
                        "remote_share": st.column_config.ProgressColumn(
                            "Remote", format="%d%%", min_value=0, max_value=100
                        ),
                    },
                )
        sources = " · ".join(f"{name} {n}" for name, n in summary["sources"].items())
        st.caption(f"Sources: {sources}. Updated daily.")
