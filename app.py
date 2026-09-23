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

st.set_page_config(page_title="JobLens", page_icon="•", layout="wide")


def api_get(path: str, **params):
    response = httpx.get(f"{API}{path}", params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


st.title("JobLens")

try:
    health = api_get("/health")
except Exception as exc:  # noqa: BLE001 - the whole page depends on this
    st.error(f"Cannot reach the API at {API}. Start it with `make serve`.\n\n{exc}")
    st.stop()

cols = st.columns(4)
cols[0].metric("postings", health["postings"])
cols[1].metric("chunks", health["chunks"])
cols[2].metric("embedder", health["embedder"] or "-")
cols[3].metric("LLM", health["llm_backend"] if health["llm_enabled"] else "off")

search_tab, chat_tab, match_tab, trends_tab = st.tabs(
    ["Search", "Chat", "Resume match", "Trends"]
)


with search_tab:
    query = st.text_input("Search", "remote machine learning engineer")
    left, middle, right = st.columns(3)
    mode = left.selectbox("mode", ["hybrid", "vector", "keyword"])
    strategy = middle.selectbox("chunking", ["whole", "section"])
    rerank = right.checkbox("cross-encoder rerank", help="slower, better ordering")

    if query:
        payload = api_get(
            "/search", q=query, mode=mode, strategy=strategy, limit=10, rerank=rerank
        )
        st.caption(f"{len(payload['results'])} results in {payload['took_ms']:.0f}ms")
        for result in payload["results"]:
            where = "remote" if result["is_remote"] else (result["location"] or "")
            st.markdown(
                f"**[{result['title']}]({result['url']})** at {result['company']}"
            )
            # Showing which retriever found it, because a search UI that
            # cannot explain a result is a search UI nobody trusts.
            st.caption(f"{where}  ·  score {result['score']:.4f}  ·  {result['ranks']}")
            if result["snippet"]:
                # ts_headline marks matches with <b>; st.text would show the tags.
                st.text(re.sub(r"</?b>", "", result["snippet"])[:300])
            st.divider()


with chat_tab:
    st.caption(
        "Answers come only from the retrieved postings. When nothing relevant "
        "is found it says so instead of guessing."
    )
    question = st.text_input(
        "Ask about the job market", "who is hiring Rust engineers?"
    )
    if st.button("Ask", type="primary"):
        with st.spinner("retrieving and answering"):
            payload = httpx.post(
                f"{API}/chat", json={"question": question, "limit": 6}, timeout=TIMEOUT
            )
        if payload.status_code == 429:
            st.warning("Rate limited. Try again in a minute.")
        elif payload.status_code == 503:
            st.warning("No LLM backend configured.")
        else:
            payload.raise_for_status()
            body = payload.json()
            st.markdown(body["answer"])
            if not body["grounded"]:
                st.warning("Not grounded: the answer cites no source.")
            for citation in body["citations"]:
                st.caption(
                    f"[{citation['n']}] [{citation['title']}]({citation['url']}) "
                    f"at {citation['company']}"
                )
            st.caption(f"{body['took_ms']}ms  ·  ${body['cost_usd']:.5f}")


with match_tab:
    st.caption(
        "Upload a resume. The useful column is what the posting wants "
        "that your resume does not mention."
    )
    upload = st.file_uploader("Resume (PDF or text)", type=["pdf", "txt", "md"])
    if upload and st.button("Match", type="primary"):
        with st.spinner("extracting skills and scoring postings"):
            response = httpx.post(
                f"{API}/match",
                files={"file": (upload.name, upload.getvalue())},
                timeout=TIMEOUT,
            )
        if response.status_code == 429:
            st.warning("Rate limited. Try again in a minute.")
        elif response.status_code in (422, 503):
            st.warning(response.json().get("detail", "could not process that file"))
        else:
            response.raise_for_status()
            body = response.json()
            st.write("**Skills found:** " + ", ".join(body["resume_skills"]))
            for match in body["matches"]:
                st.markdown(
                    f"**{match['fit_score']}/100 · "
                    f"[{match['title']}]({match['url']})** at {match['company']}"
                )
                st.write(match["reasoning"])
                st.caption("matched: " + (", ".join(match["matched_skills"]) or "none"))
                st.caption("missing: " + (", ".join(match["missing_skills"]) or "none"))
                st.divider()
            st.caption(f"{body['took_ms']}ms  ·  ${body['cost_usd']:.5f}")


with trends_tab:
    days = st.slider("window (days)", 7, 365, 90)
    summary = api_get("/trends", days=days)
    if not summary.get("postings"):
        st.info("No postings in this window.")
    else:
        row = st.columns(4)
        row[0].metric("postings", summary["postings"])
        row[1].metric("remote", f"{summary['remote_share']:.0%}")
        row[2].metric("state a salary", f"{summary['salary_coverage']:.0%}")
        row[3].metric(
            "median salary",
            (
                f"${summary['median_salary_usd']:,}"
                if summary["median_salary_usd"]
                else "-"
            ),
        )
        skills = pd.DataFrame(summary["top_skills"])
        if not skills.empty:
            st.bar_chart(skills.set_index("skill")["postings"])
        regions = pd.DataFrame(summary["top_regions"])
        if not regions.empty:
            st.dataframe(regions, hide_index=True, use_container_width=True)
