"""The reranker's Phase 3 options, with a fake cross-encoder so no model loads."""

import pytest

from joblens.eval.retrieval import _percentile
from joblens.search import rerank
from joblens.search.retrieval import SearchHit


def hit(posting_id, score, title="Data Scientist", snippet="snippet text"):
    return SearchHit(
        posting_id=posting_id,
        title=title,
        company="Acme",
        url="",
        location="Pune",
        is_remote=False,
        score=score,
        snippet=snippet,
    )


class FakeModel:
    """Scores a pair by a number found in the document, and records inputs."""

    def __init__(self, scores):
        self.scores = scores
        self.pairs = []

    def predict(self, pairs, show_progress_bar=False):
        self.pairs.extend(pairs)
        return [self.scores[i] for i in range(len(pairs))]


@pytest.fixture
def fake(monkeypatch):
    def install(scores):
        model = FakeModel(scores)
        monkeypatch.setattr(rerank, "_load", lambda *a, **k: model)
        return model

    return install


def test_default_reorders_by_cross_encoder_alone(fake):
    fake([0.1, 0.9, 0.5])
    hits = [hit(1, 3.0), hit(2, 2.0), hit(3, 1.0)]
    ranked = rerank.rerank_hits("q", hits, limit=3)
    assert [h.posting_id for h in ranked] == [2, 3, 1]
    # The first-stage rank survives, for "did reranking help" afterwards.
    assert ranked[0].ranks["first_stage"] == 2


def test_alpha_blends_in_the_first_stage(fake):
    # First stage strongly prefers 1; the cross-encoder mildly prefers 2.
    fake([0.40, 0.50, 0.0])
    hits = [hit(1, 1.0), hit(2, 0.1), hit(3, 0.0)]
    pure = rerank.rerank_hits("q", [*hits], limit=3)
    assert pure[0].posting_id == 2
    fake([0.40, 0.50, 0.0])
    hits = [hit(1, 1.0), hit(2, 0.1), hit(3, 0.0)]
    blended = rerank.rerank_hits(
        "q", hits, limit=3, config=rerank.RerankConfig(alpha=0.5)
    )
    assert blended[0].posting_id == 1


def test_alpha_one_keeps_the_first_stage_order(fake):
    fake([0.9, 0.1, 0.5])
    hits = [hit(1, 3.0), hit(2, 2.0), hit(3, 1.0)]
    ranked = rerank.rerank_hits(
        "q", hits, limit=3, config=rerank.RerankConfig(alpha=1.0)
    )
    assert [h.posting_id for h in ranked] == [1, 2, 3]


def test_the_pool_defaults_to_three_times_the_limit():
    assert rerank.RerankConfig().pool(10) == 30
    assert rerank.RerankConfig(candidates=50).pool(10) == 50


def test_bad_settings_are_refused():
    with pytest.raises(ValueError):
        rerank.RerankConfig(document="everything")
    with pytest.raises(ValueError):
        rerank.RerankConfig(alpha=1.5)


def test_snippet_document_reads_title_company_and_snippet(fake):
    model = fake([0.0])
    rerank.rerank_hits("q", [hit(1, 1.0, snippet="builds models")], limit=1)
    ((_, document),) = model.pairs
    assert document == "Data Scientist. Acme. Pune. builds models"


def test_head_document_reads_the_description(fake, monkeypatch):
    model = fake([0.0])
    monkeypatch.setattr(
        rerank, "_texts", lambda conn, q, hits, config: ["the full head text"]
    )
    rerank.rerank_hits(
        "q", [hit(1, 1.0)], limit=1, config=rerank.RerankConfig(document="head")
    )
    ((_, document),) = model.pairs
    assert document.endswith("the full head text")


def test_normalised_handles_a_flat_list():
    assert rerank._normalised([2.0, 2.0]) == [0.0, 0.0]
    assert rerank._normalised([1.0, 3.0, 2.0]) == [0.0, 1.0, 0.5]


@pytest.mark.parametrize(
    "values, q, expected",
    [
        ([], 95, 0.0),
        ([5.0], 95, 5.0),
        (list(range(1, 21)), 95, 19),  # nearest rank: ceil(0.95 * 20) = 19th
        (list(range(1, 21)), 50, 10),
        ([10, 1, 7, 3], 50, 3),
    ],
)
def test_percentile_is_nearest_rank(values, q, expected):
    assert _percentile(values, q) == expected
