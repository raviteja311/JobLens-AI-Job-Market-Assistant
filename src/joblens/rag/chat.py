"""RAG over the postings: retrieve, ground, cite, and refuse when empty.

The interesting part of this file is `MIN_SIMILARITY` and what happens below it.
A RAG system that always answers is easy to build and useless: ask it
something the corpus cannot answer and it will produce a fluent paragraph
about the job market in general, which is exactly the output a user cannot
tell apart from a real one. So retrieval decides whether there is an answer
at all, and when it says no, the model is never called.

That threshold is the honest-failure switch, and it is the first thing to
check when someone reports a hallucination.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from joblens.cleaning import strip_noise
from joblens.config import get_settings
from joblens.llm import client, prompts
from joblens.search import retrieval
from joblens.search.embeddings import Embedder

log = logging.getLogger(__name__)

# Below this cosine similarity between the question and its best-matching
# posting, nothing retrieved is worth answering from and the model is not
# called.
#
# It is a raw vector similarity on purpose. The gate used to be a hybrid RRF
# score of 0.016, but a single retriever's rank-1 hit alone scores 1/61 =
# 0.0164 and vector search always returns neighbours, so every question in
# data/golden/chat.yaml passed it, "what is the capital of Peru?" included.
# Measured on that set (466 postings, whole-posting vectors): the 15
# answerable questions have a best similarity of 0.408 to 0.633; the 5 that
# must be refused have 0.163 to 0.421. No threshold separates them all
# (the Elon Musk question sits inside the answerable range), so 0.30 is set
# to catch the clearly off-topic with a wide margin and never refuse a real
# question. Near-topic questions the corpus cannot answer are left to the
# model's own refusal, which the citation check then catches.
MIN_SIMILARITY = 0.30

NO_ANSWER = (
    "I could not find postings that answer that. This corpus is {n} job "
    "postings from Hacker News and RemoteOK, so it will not know about a "
    "company or a technology that none of them mention."
)


def no_answer(conn) -> str:
    """The refusal, with the real corpus size rather than a number that was
    true the week it was typed."""
    try:
        n = conn.execute("select count(*) as n from postings").fetchone()["n"]
    except Exception:  # noqa: BLE001 - the refusal must not fail on a count
        n = "a few hundred"
    return NO_ANSWER.format(n=n)


# How much of each posting the model gets to read. Enough for requirements,
# short enough that eight sources fit in a small local model's context.
SOURCE_CHARS = 900


@dataclass
class Source:
    n: int
    posting_id: int
    title: str
    company: str
    url: str
    location: str | None
    text: str
    is_remote: bool = False

    def render(self) -> str:
        # Only say "remote" when the posting says so. A missing location is
        # "not stated", and telling the model otherwise is a hallucination
        # we would be feeding it ourselves.
        where = self.location or ("remote" if self.is_remote else "")
        head = f"[{self.n}] {self.title} at {self.company}"
        if where:
            head += f" ({where})"
        return f"{head}\n{self.text}"


@dataclass
class ChatAnswer:
    question: str
    answer: str
    sources: list[Source]
    grounded: bool
    took_ms: int = 0
    cost_usd: float = 0.0
    cited: set[int] = field(default_factory=set)


_CITATION = re.compile(r"\[(\d+)\]")


def collect_sources(conn, question: str, embedder: Embedder | None, limit: int):
    strategy = get_settings().chunk_strategy
    best = retrieval.vector_search(
        conn, question, embedder=embedder, strategy=strategy, limit=1
    )
    if not best or best[0].score < MIN_SIMILARITY:
        return []
    hits = retrieval.search(
        conn,
        question,
        mode="hybrid",
        embedder=embedder,
        strategy=strategy,
        limit=limit,
    )
    if not hits:
        return []
    rows = conn.execute(
        "select id, description from postings where id = any(%s)",
        ([h.posting_id for h in hits],),
    ).fetchall()
    bodies = {r["id"]: strip_noise(r["description"] or "") for r in rows}
    return [
        Source(
            n=i,
            posting_id=hit.posting_id,
            title=hit.title,
            company=hit.company,
            url=hit.url,
            location=hit.location,
            text=bodies.get(hit.posting_id, "")[:SOURCE_CHARS],
            is_remote=hit.is_remote,
        )
        for i, hit in enumerate(hits, start=1)
    ]


def ask(
    conn,
    question: str,
    embedder: Embedder | None = None,
    limit: int = 8,
    prompt_version: str | None = None,
) -> ChatAnswer:
    began = time.perf_counter()
    sources = collect_sources(conn, question, embedder, limit)

    if not sources:
        # No model call at all. Cheaper, faster, and it cannot hallucinate.
        return ChatAnswer(
            question=question,
            answer=no_answer(conn),
            sources=[],
            grounded=False,
            took_ms=int((time.perf_counter() - began) * 1000),
        )

    prompt = prompts.load("chat_answer", prompt_version)
    # Fenced and stripped of fence tags: the postings are scraped text and
    # the question is user text, and neither gets to rewrite the rules.
    rendered = prompt.render(
        sources=prompts.untagged("\n\n".join(s.render() for s in sources)),
        question=prompts.untagged(question),
    )
    completion = client.complete(
        rendered, feature="chat", prompt=prompt, max_tokens=600
    )

    cited = {int(n) for n in _CITATION.findall(completion.text)}
    # A citation pointing at a source that was never supplied is the model
    # inventing a reference. Flagged rather than silently dropped.
    invalid = cited - {s.n for s in sources}
    if invalid:
        log.warning("answer cited sources that were not supplied: %s", sorted(invalid))

    return ChatAnswer(
        question=question,
        answer=completion.text.strip(),
        sources=sources,
        grounded=bool(cited) and not invalid,
        took_ms=int((time.perf_counter() - began) * 1000),
        cost_usd=completion.cost_usd,
        cited=cited,
    )
