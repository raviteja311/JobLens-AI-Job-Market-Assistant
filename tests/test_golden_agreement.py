"""scripts/golden_agreement.py produces the kappa the README quotes."""

import importlib.util
import json
from pathlib import Path

import pytest

from joblens.eval import golden

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "golden_agreement.py"


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location("golden_agreement", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_identical_grades_agree_perfectly(script, tmp_path):
    grades = {"q01": {"a:1": 0, "a:2": 1, "a:3": 2}}
    a = script.load_grades(write(tmp_path, "a.json", grades))
    result = script.agreement(script.paired(a, a))
    assert result["exact"] == 1.0
    assert result["kappa_linear"] == pytest.approx(1.0)


def test_only_pairs_graded_on_both_sides_count(script, tmp_path):
    a = script.load_grades(
        write(tmp_path, "a.json", {"q01": {"a:1": 2, "a:2": 0}, "q02": {"b:1": 1}})
    )
    # A half-filled template: nulls are ungraded, not zeros.
    b = script.load_grades(write(tmp_path, "b.json", {"q01": {"a:1": 2, "a:2": None}}))
    assert script.paired(a, b) == [("q01", "a:1", 2, 2)]


def test_a_near_miss_costs_less_than_a_far_miss(script):
    truth = [("q", f"k{i}", g, g) for i, g in enumerate([0, 0, 1, 1, 2, 2])]
    near = [(q, k, x, 1 if x == 2 else y) for q, k, x, y in truth]
    far = [(q, k, x, 0 if x == 2 else y) for q, k, x, y in truth]
    assert script.agreement(near)["exact"] == script.agreement(far)["exact"]
    assert (
        script.agreement(near)["kappa_linear"] > script.agreement(far)["kappa_linear"]
    )


def test_reads_the_golden_yaml(script, tmp_path):
    path = tmp_path / "retrieval.yaml"
    golden.save([golden.GoldenQuery(id="q01", query="x", judgements={"a:1": 2})], path)
    assert script.load_grades(path) == {"q01": {"a:1": 2}}


def test_main_reports(script, tmp_path, capsys):
    a = write(tmp_path, "a.json", {"q01": {"a:1": 2, "a:2": 0, "a:3": 1}})
    b = write(tmp_path, "b.json", {"q01": {"a:1": 2, "a:2": 1, "a:3": 1}})
    assert script.main([str(a), str(b), "--show", "5"]) == 0
    out = capsys.readouterr().out
    assert "3 pairs over 1 queries" in out
    assert "q01 a:2: 0 vs 1" in out
