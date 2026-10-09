"""Cross-encoder reranking of the top candidates.

A bi-encoder embeds the query and the posting separately and compares the two
vectors, which is what makes it fast enough to search the whole corpus: the
postings were embedded last night. The cost is that the model never sees the
query and the posting at the same time, so it cannot notice that the query
said "no PhD required" and the posting demands one.

A cross-encoder reads both together and scores the pair. It cannot be indexed
and has to run per candidate, so it only ever sees the top few results from a
cheap retriever. By default the first stage returns `limit * 3` candidates (30
for the API's default of 10) and the second stage orders those.

It can only reorder what it is given. A relevant posting the first stage did
not return is lost however good the cross-encoder is, which is why reranking
can lower recall@10: it promotes a different set into the top ten, and nothing
outside the pool can be promoted back. RerankConfig exposes the levers the
Phase 3 experiments pull (docs/experiments.md): the pool size, what text the
model reads, blending in the first-stage score, and a cheaper model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# About 400 tokens of English. The ms-marco cross-encoders were trained on
# passages, and the requirements usually start well below a posting's first
# paragraph, so the head is a compromise rather than the whole posting.
HEAD_CHARS = 1600

DOCUMENTS = ("snippet", "head", "section")


@dataclass(frozen=True)
class RerankConfig:
    """How the second stage runs. The defaults are what shipped before Phase 3.

    model       which cross-encoder
    document    what it reads per candidate: "snippet" (the first stage's
                snippet), "head" (title, company, location and the first
                HEAD_CHARS of the description) or "section" (the posting's
                section chunk closest to the query)
    max_length  token cap on query plus document; None keeps the model's own
    alpha       None reorders by cross-encoder score alone. A number blends:
                alpha * first-stage score + (1 - alpha) * rerank score, both
                min-max normalised over the pool, so the first stage's
                judgement is not thrown away
    candidates  pool size handed to the cross-encoder; None means limit * 3
    """

    model: str = CROSS_ENCODER_MODEL
    document: str = "snippet"
    max_length: int | None = None
    alpha: float | None = None
    candidates: int | None = None

    def __post_init__(self) -> None:
        if self.document not in DOCUMENTS:
            raise ValueError(f"document must be one of {DOCUMENTS}")
        if self.alpha is not None and not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")

    def pool(self, limit: int) -> int:
        return self.candidates if self.candidates else limit * 3


DEFAULT = RerankConfig()


@dataclass
class RerankUsage:
    pairs: int = 0
    seconds: float = 0.0

    @property
    def ms_per_pair(self) -> float:
        return (self.seconds * 1000 / self.pairs) if self.pairs else 0.0


usage: RerankUsage = RerankUsage()

_models: dict[tuple[str, int | None], object] = {}


def _load(model: str = CROSS_ENCODER_MODEL, max_length: int | None = None):
    key = (model, max_length)
    if key not in _models:
        from sentence_transformers import CrossEncoder

        log.info("loading %s (max_length=%s)", model, max_length)
        kwargs = {"max_length": max_length} if max_length else {}
        _models[key] = CrossEncoder(model, **kwargs)
    return _models[key]


_embedder = None


def _query_embedder():
    """One embedder for the "section" document, kept for the process.
    get_embedder() builds a new one per call, and a new LocalEmbedder loads
    its model again, which would bill every reranked query a model load."""
    global _embedder
    if _embedder is None:
        from joblens.search.embeddings import get_embedder

        _embedder = get_embedder()
    return _embedder


def _header(hit) -> list[str]:
    return [hit.title, hit.company, hit.location or ""]


def _document(hit, text: str | None = None) -> str:
    """What the cross-encoder reads for a candidate: who and where, then
    either the first stage's snippet or the text the config asked for."""
    body = hit.snippet if text is None else text
    return ". ".join(part for part in [*_header(hit), body] if part)[:2000]


def _texts(conn, query: str, hits: list, config: RerankConfig) -> list[str | None]:
    """The body text per hit for the configured document, None for snippet."""
    if config.document == "snippet" or conn is None:
        return [None] * len(hits)
    ids = [hit.posting_id for hit in hits]
    if config.document == "head":
        rows = conn.execute(
            "select id, left(description, %s) as body from postings"
            " where id = any(%s)",
            (HEAD_CHARS, ids),
        ).fetchall()
    else:
        embedder = _query_embedder()
        vector = embedder.encode([query])[0]
        literal = "[" + ",".join(f"{v:.6f}" for v in vector) + "]"
        rows = conn.execute(
            """
            select distinct on (posting_id) posting_id as id, content as body
              from posting_chunks
             where strategy = 'section' and model = %s and posting_id = any(%s)
             order by posting_id, embedding <=> %s::vector
            """,
            (embedder.name, ids, literal),
        ).fetchall()
    found = {row["id"]: row["body"] for row in rows}
    return [found.get(hit.posting_id) for hit in hits]


def _normalised(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if high == low:
        return [0.0 for _ in values]
    return [(v - low) / (high - low) for v in values]


def rerank_hits(
    query: str,
    hits: list,
    limit: int = 20,
    config: RerankConfig = DEFAULT,
    conn=None,
) -> list:
    """Re-order candidates by cross-encoder score. Returns the same objects.

    `conn` is needed only for the "head" and "section" documents, which read
    text the first stage did not return.
    """
    if not hits:
        return hits
    model = _load(config.model, config.max_length)
    texts = _texts(conn, query, hits, config)
    pairs = [
        (query, _document(hit, text)) for hit, text in zip(hits, texts, strict=True)
    ]
    first_stage = [hit.score for hit in hits]
    began = time.perf_counter()
    scores = [float(s) for s in model.predict(pairs, show_progress_bar=False)]
    usage.pairs += len(pairs)
    usage.seconds += time.perf_counter() - began

    if config.alpha is not None:
        scores = [
            config.alpha * f + (1 - config.alpha) * r
            for f, r in zip(_normalised(first_stage), _normalised(scores), strict=True)
        ]
    for rank, (hit, score) in enumerate(zip(hits, scores, strict=True), start=1):
        # The first-stage rank is kept, because losing it would make the
        # "did reranking help" question unanswerable after the fact.
        hit.ranks["first_stage"] = rank
        hit.score = score
    return sorted(hits, key=lambda h: h.score, reverse=True)[:limit]
