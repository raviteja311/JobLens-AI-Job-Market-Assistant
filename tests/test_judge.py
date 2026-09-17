"""Cohen's kappa, which is the only reason the judge scores mean anything."""

import pytest

from joblens.eval.judge import Calibration, _kappa


def test_kappa_is_zero_for_a_judge_that_always_says_two():
    human = [2, 2, 2, 1, 0, 2, 2, 1]
    machine = [2] * 8
    # Raw agreement looks respectable and kappa does not, which is the whole
    # reason the calibration reports both.
    agreement = sum(1 for h, m in zip(human, machine, strict=True) if h == m) / len(
        human
    )
    assert agreement > 0.5
    assert _kappa(human, machine) == pytest.approx(0.0, abs=1e-9)


def test_kappa_is_one_for_perfect_agreement():
    assert _kappa([0, 1, 2, 2], [0, 1, 2, 2]) == pytest.approx(1.0)


def test_kappa_of_nothing_is_zero():
    assert _kappa([], []) == 0.0


def test_an_uncalibrated_judge_is_not_trustworthy():
    # Fewer than 20 hand-scored answers, so the numbers are a smoke test.
    assert not Calibration(0, 0.0, 0.0, 0.0, 0.0).trustworthy
    assert not Calibration(8, 1.0, 1.0, 1.0, 1.0).trustworthy
    assert Calibration(20, 0.8, 0.55, 0.8, 0.5).trustworthy


def test_calibration_table_says_when_it_is_not_usable():
    assert "NOT usable" in Calibration(4, 1.0, 1.0, 1.0, 1.0).as_table()
