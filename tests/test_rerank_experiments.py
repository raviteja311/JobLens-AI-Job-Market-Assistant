"""The decision rule in scripts/rerank_experiments.py, on made-up scores."""

import importlib.util
from pathlib import Path

import pytest

from joblens.eval.retrieval import ConfigScore, RetrievalReport

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rerank_experiments.py"


@pytest.fixture
def rx():
    spec = importlib.util.spec_from_file_location("rerank_experiments", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(name, mrr, p95):
    return ConfigScore(
        name=name, scores={"mrr": mrr}, queries=1, seconds=0.0, latencies_ms=[p95]
    )


def report(rx, *configs):
    return RetrievalReport(
        configs=[config(rx.BASELINE, 0.60, 200), *configs], queries=1, corpus=1
    )


def test_highest_mrr_wins_then_lowest_p95(rx):
    passing = [config("a", 0.70, 400), config("b", 0.72, 450), config("c", 0.72, 300)]
    assert rx.winner(passing).name == "c"
    assert rx.winner([]) is None


def test_a_slow_or_small_gain_does_not_pass(rx):
    text = rx.verdict(
        report(rx, config("slow", 0.80, 2000), config("small", 0.64, 300)),
        provisional=False,
    )
    assert "meeting the rule (MRR >= +0.05, p95 < 500 ms): none" in text
    assert "hybrid stays the default" in text


def test_the_winner_is_named_in_the_decision(rx):
    text = rx.verdict(
        report(rx, config("fast", 0.70, 300), config("faster", 0.70, 250)),
        provisional=False,
    )
    assert "decision: rerank ON by default, configured as faster." in text


def test_a_provisional_run_decides_nothing(rx):
    text = rx.verdict(report(rx, config("fast", 0.70, 300)), provisional=True)
    assert "PROVISIONAL" in text
    assert "decision:" not in text
