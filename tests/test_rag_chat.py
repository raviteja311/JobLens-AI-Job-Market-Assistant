"""The chat relevance gate, without a database or a model.

The gate decides whether the LLM is called at all. It used to compare the
fused hybrid score with 0.016, which a single retriever's rank-1 hit always
clears, so these tests pin down that it now reads the raw vector similarity.
"""

from __future__ import annotations

from joblens.rag import chat
from joblens.search.retrieval import SearchHit


def _hit(posting_id: int, score: float) -> SearchHit:
    return SearchHit(
        posting_id=posting_id,
        title=f"Role {posting_id}",
        company="Acme",
        url=f"https://example.com/{posting_id}",
        location=None,
        is_remote=True,
        score=score,
    )


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class _Conn:
    def execute(self, sql, params=None):
        return _Rows([{"id": 1, "description": "Rust and Postgres, remote."}])


def _patch(monkeypatch, best_similarity: float):
    calls = {"hybrid": 0}

    def vector_search(conn, question, embedder=None, strategy="whole", limit=20):
        return [_hit(1, best_similarity)]

    def search(conn, question, **kwargs):
        calls["hybrid"] += 1
        # A fused score well above the old 0.016 gate, as rank 1 always is.
        return [_hit(1, 1 / 61)]

    monkeypatch.setattr(chat.retrieval, "vector_search", vector_search)
    monkeypatch.setattr(chat.retrieval, "search", search)
    return calls


def test_an_off_topic_question_gets_no_sources(monkeypatch):
    # "what is the capital of Peru?" measured 0.163 against the corpus.
    calls = _patch(monkeypatch, best_similarity=0.163)
    assert chat.collect_sources(_Conn(), "what is the capital of Peru?", None, 6) == []
    assert calls["hybrid"] == 0


def test_a_relevant_question_gets_numbered_sources(monkeypatch):
    calls = _patch(monkeypatch, best_similarity=0.558)
    sources = chat.collect_sources(_Conn(), "who is hiring Rust engineers?", None, 6)
    assert calls["hybrid"] == 1
    assert [(s.n, s.posting_id) for s in sources] == [(1, 1)]
    assert sources[0].text.startswith("Rust and Postgres")


def test_the_gate_sits_below_every_answerable_golden_question():
    # The lowest answerable question in data/golden/chat.yaml scored 0.408.
    assert chat.MIN_SIMILARITY < 0.408
