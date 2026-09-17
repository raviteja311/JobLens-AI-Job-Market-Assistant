"""Cross-encoder reranking of the top candidates.

A bi-encoder embeds the query and the posting separately and compares the two
vectors, which is what makes it fast enough to search the whole corpus: the
postings were embedded last night. The cost is that the model never sees the
query and the posting at the same time, so it cannot notice that the query
said "no PhD required" and the posting demands one.

A cross-encoder reads both together and scores the pair. It cannot be indexed
and has to run per candidate, so it only ever sees the top few results from a
cheap retriever. First stage picks 60 from 465, second stage orders those 60.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None


@dataclass
class RerankUsage:
    pairs: int = 0
    seconds: float = 0.0

    @property
    def ms_per_pair(self) -> float:
        return (self.seconds * 1000 / self.pairs) if self.pairs else 0.0


usage: RerankUsage = RerankUsage()


def _load():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        log.info("loading %s", CROSS_ENCODER_MODEL)
        _model = CrossEncoder(CROSS_ENCODER_MODEL)
    return _model


def _document(hit) -> str:
    """What the cross-encoder reads for a candidate.

    The snippet rather than the whole description: the model truncates at 512
    tokens anyway, and the snippet is the part the first stage already thought
    was relevant.
    """
    parts = [hit.title, hit.company, hit.location or "", hit.snippet]
    return ". ".join(part for part in parts if part)[:2000]


def rerank_hits(query: str, hits: list, limit: int = 20) -> list:
    """Re-order candidates by cross-encoder score. Returns the same objects."""
    if not hits:
        return hits
    model = _load()
    pairs = [(query, _document(hit)) for hit in hits]
    began = time.perf_counter()
    scores = model.predict(pairs, show_progress_bar=False)
    usage.pairs += len(pairs)
    usage.seconds += time.perf_counter() - began

    for rank, (hit, score) in enumerate(zip(hits, scores, strict=True), start=1):
        # The first-stage rank is kept, because losing it would make the
        # "did reranking help" question unanswerable after the fact.
        hit.ranks["first_stage"] = rank
        hit.score = float(score)
    return sorted(hits, key=lambda h: h.score, reverse=True)[:limit]
