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


# ---------------------------------------------------------------- the gate


GOOD = {"ndcg@10": 0.62, "recall@10": 0.66, "mrr": 0.76}


def _no_history(monkeypatch):
    from joblens.eval import report

    monkeypatch.setattr(report, "previous", lambda suite, config: None)
    return report


def test_the_gate_falls_back_to_the_committed_baseline(monkeypatch, tmp_path):
    # CI starts from an empty database, so there is never a last run there.
    # Without the committed file only the floors applied.
    report = _no_history(monkeypatch)
    baseline = tmp_path / "eval_baseline.json"
    monkeypatch.setattr(report, "BASELINE_JSON", baseline)
    report.write_baseline("retrieval", {"vector (whole)": GOOD}, queries=58)

    dropped = {**GOOD, "mrr": 0.60}  # above every floor, 0.16 below baseline
    gate = report.gate("retrieval", "vector (whole)", dropped)
    assert not gate.passed
    assert "mrr dropped 0.160" in gate.failures[0]
    assert report.gate("retrieval", "vector (whole)", GOOD).passed


def test_the_last_recorded_run_beats_the_committed_baseline(monkeypatch, tmp_path):
    from joblens.eval import report

    monkeypatch.setattr(report, "BASELINE_JSON", tmp_path / "eval_baseline.json")
    report.write_baseline("retrieval", {"keyword": GOOD})
    monkeypatch.setattr(report, "previous", lambda suite, config: {"mrr": 0.5})
    assert report.baseline_for("retrieval", "keyword") == {"mrr": 0.5}


def test_uncalibrated_judge_scores_are_not_regression_gated(monkeypatch):
    report = _no_history(monkeypatch)
    before = {"refusal_accuracy": 1.0, "faithfulness": 0.9, "completeness": 0.9}
    after = {"refusal_accuracy": 1.0, "faithfulness": 0.5, "completeness": 0.5}
    assert report.gate("chat", "v2", after, baseline=before).passed
    after["refusal_accuracy"] = 0.8
    assert not report.gate("chat", "v2", after, baseline=before).passed


def test_the_committed_baseline_covers_every_retrieval_configuration():
    from joblens.eval import report
    from joblens.eval.retrieval import CONFIGURATIONS

    for config in CONFIGURATIONS:
        scores = report.committed_baseline("retrieval", config["name"])
        assert scores, f"no committed baseline for {config['name']}"
        assert {"ndcg@10", "recall@10", "mrr"} <= set(scores)


def test_eval_compares_against_the_baseline_from_before_this_run(monkeypatch):
    # `record` used to run first, so "the last run" was this run and the
    # regression check compared a run with itself.
    from types import SimpleNamespace

    from joblens import cli
    from joblens.eval import report
    from joblens.eval import retrieval as eval_retrieval

    stored: list[dict] = [{**GOOD, "mrr": 0.90}]
    config = SimpleNamespace(name="vector (whole)", scores=GOOD)
    result = SimpleNamespace(
        configs=[config], best=config, queries=58, corpus=466, as_table=lambda: ""
    )
    monkeypatch.setattr(eval_retrieval, "run", lambda: result)
    monkeypatch.setattr(report, "previous", lambda suite, name: stored[-1])
    monkeypatch.setattr(
        report, "record", lambda suite, name, n, scores: stored.append(scores)
    )
    args = SimpleNamespace(suite="retrieval", strict=True, update_baseline=False)
    assert cli.cmd_eval(args) == 1  # mrr fell 0.14 against the earlier run
