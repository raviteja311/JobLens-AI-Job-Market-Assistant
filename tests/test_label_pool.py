"""The labelling pool covers every configuration the eval scores."""

from joblens.eval import label


class FakeConn:
    def execute(self, sql, params):
        return self

    def fetchall(self):
        return []


def test_pool_draws_from_hybrid_three_times_as_deep(monkeypatch):
    calls = []

    def fake_search(conn, query, mode, embedder, strategy, limit):
        calls.append((mode, strategy, limit))
        return []

    monkeypatch.setattr(label.retrieval, "search", fake_search)
    assert label.pool(FakeConn(), "data engineer Pune", depth=10) == []
    assert calls == [
        ("keyword", "whole", 10),
        ("vector", "whole", 10),
        ("vector", "section", 10),
        # The default search, as deep as the reranker's pool, so every
        # reranked top ten is judged.
        ("hybrid", "whole", 30),
        ("hybrid", "section", 10),
    ]
