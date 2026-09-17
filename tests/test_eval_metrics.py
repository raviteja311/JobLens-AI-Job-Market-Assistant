import pytest

from joblens.eval import metrics
from joblens.search.retrieval import RRF_K, SearchHit, reciprocal_rank_fusion

RELEVANT = {10: 2, 11: 1, 12: 2}


def test_recall_counts_graded_relevance_as_relevant():
    assert metrics.recall_at_k([10, 11, 12], RELEVANT, 5) == 1.0
    assert metrics.recall_at_k([10, 99, 98], RELEVANT, 5) == pytest.approx(1 / 3)
    assert metrics.recall_at_k([99], RELEVANT, 5) == 0.0


def test_recall_at_k_respects_the_cutoff():
    assert metrics.recall_at_k([99, 98, 97, 96, 95, 10], RELEVANT, 5) == 0.0
    assert metrics.recall_at_k([99, 98, 97, 96, 95, 10], RELEVANT, 10) == pytest.approx(
        1 / 3
    )


def test_reciprocal_rank_is_the_first_hit():
    assert metrics.reciprocal_rank([10], RELEVANT) == 1.0
    assert metrics.reciprocal_rank([99, 10], RELEVANT) == 0.5
    assert metrics.reciprocal_rank([99, 98], RELEVANT) == 0.0


def test_ndcg_rewards_putting_the_best_first():
    good = metrics.ndcg_at_k([10, 12, 11], RELEVANT, 10)
    bad = metrics.ndcg_at_k([11, 10, 12], RELEVANT, 10)
    assert good == 1.0
    assert bad < good


def test_ndcg_of_nothing_relevant_is_zero():
    assert metrics.ndcg_at_k([99, 98], RELEVANT, 10) == 0.0
    assert metrics.ndcg_at_k([10], {}, 10) == 0.0


def test_recall_and_mrr_can_disagree():
    # The case the three-metric report exists for: this run found one more
    # relevant posting than the other, and buried it.
    shallow = [10, 99, 99, 99, 99]
    deep = [99, 99, 99, 99, 10, 11]
    assert metrics.reciprocal_rank(shallow, RELEVANT) > metrics.reciprocal_rank(
        deep, RELEVANT
    )
    assert metrics.recall_at_k(deep, RELEVANT, 10) > metrics.recall_at_k(
        shallow, RELEVANT, 10
    )


def test_mean_scores_averages_every_key():
    mean = metrics.mean_scores([{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}])
    assert mean == {"a": 0.5, "b": 0.5}
    assert metrics.mean_scores([]) == {}


def _hit(posting_id: int) -> SearchHit:
    return SearchHit(
        posting_id=posting_id,
        title=f"Job {posting_id}",
        company="Co",
        url="https://example.com",
        location=None,
        is_remote=True,
        score=0.0,
    )


def test_rrf_rewards_appearing_in_both_runs():
    runs = {
        "keyword": [_hit(1), _hit(2), _hit(3)],
        "vector": [_hit(3), _hit(4), _hit(5)],
    }
    merged = reciprocal_rank_fusion(runs, limit=5)
    # 3 is rank 3 and rank 1; nothing else is in both.
    assert merged[0].posting_id == 3
    assert merged[0].ranks == {"keyword": 3, "vector": 1}


def test_rrf_score_matches_the_formula():
    merged = reciprocal_rank_fusion({"keyword": [_hit(1)]}, limit=1)
    assert merged[0].score == pytest.approx(1 / (RRF_K + 1))


def test_rrf_ignores_how_confident_each_retriever_was():
    # Different raw scores, same ranks, so the fusion must not change.
    a, b = _hit(1), _hit(2)
    a.score, b.score = 99.0, 0.001
    first = reciprocal_rank_fusion({"keyword": [a, b]}, limit=2)
    a.score, b.score = 0.001, 99.0
    second = reciprocal_rank_fusion({"keyword": [a, b]}, limit=2)
    assert [h.posting_id for h in first] == [h.posting_id for h in second]
