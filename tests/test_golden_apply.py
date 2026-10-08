"""scripts/golden_apply.py writes the file every retrieval number is scored
against, so a merge that loses zeros or stacks provenance notes is a silent
change to the leaderboard."""

import importlib.util
import json
from pathlib import Path

import pytest

from joblens.eval import golden

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "golden_apply.py"


@pytest.fixture
def apply(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("golden_apply", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = tmp_path / "retrieval.yaml"
    monkeypatch.setattr(golden, "RETRIEVAL_FILE", target)
    golden.save(
        [
            golden.GoldenQuery(
                id="q01",
                query="remote llm roles",
                judgements={"remoteok:1": 2},
                note="rare-token case | judged by an older pass",
            )
        ]
    )
    return module, tmp_path


def _grades(tmp_path, payload, name="grades.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_merge_keeps_zeros_and_adds_its_provenance(apply):
    module, tmp = apply
    grades = _grades(tmp, {"q01": {"hackernews:9": 0, "remoteok:2": 1}})
    assert (
        module.main([grades, "--grader", "human", "--note", "judged by a reviewer"])
        == 0
    )
    (query,) = golden.load()
    assert query.judgements == {"remoteok:1": 2, "hackernews:9": 0, "remoteok:2": 1}
    # The fixture's older grades were never human-verified, so a human top-up
    # on top of them cannot make every judgement a human's.
    assert not query.verified
    assert query.note == (
        "rare-token case | judged by an older pass | judged by a reviewer"
    )
    # Applying the same batch twice does not stack the note.
    module.main([grades, "--grader", "human", "--note", "judged by a reviewer"])
    assert golden.load()[0].note.count("judged by a reviewer") == 1


def test_replace_drops_old_judgements(apply):
    module, tmp = apply
    grades = _grades(tmp, {"q01": {"remoteok:2": 2}})
    module.main(
        [grades, "--grader", "human", "--note", "judged by a reviewer", "--replace"]
    )
    (query,) = golden.load()
    assert query.judgements == {"remoteok:2": 2}
    assert query.note == "rare-token case | judged by a reviewer"


def test_out_of_range_grade_is_refused_before_anything_is_written(apply):
    module, tmp = apply
    grades = _grades(tmp, {"q01": {"remoteok:2": 3}})
    with pytest.raises(SystemExit, match="0, 1 or 2"):
        module.main([grades, "--grader", "human", "--note", "x"])
    assert golden.load()[0].judgements == {"remoteok:1": 2}


def test_unknown_query_ids_are_skipped(apply, capsys):
    module, tmp = apply
    module.main(
        [_grades(tmp, {"q99": {"remoteok:2": 1}}), "--grader", "human", "--note", "x"]
    )
    assert "skipping unknown query q99" in capsys.readouterr().out


def test_a_human_replace_verifies_the_query(apply):
    module, tmp = apply
    grades = _grades(tmp, {"q01": {"remoteok:2": 2}})
    module.main([grades, "--grader", "human", "--note", "by me", "--replace"])
    assert golden.load()[0].verified


def test_llm_grades_never_verify(apply):
    module, tmp = apply
    grades = _grades(tmp, {"q01": {"remoteok:2": 2}})
    module.main([grades, "--grader", "llm", "--note", "by a model", "--replace"])
    assert not golden.load()[0].verified


def test_an_llm_top_up_unverifies_a_human_query(apply):
    module, tmp = apply
    human = _grades(tmp, {"q01": {"remoteok:2": 2}}, "human.json")
    module.main([human, "--grader", "human", "--note", "by me", "--replace"])
    assert golden.load()[0].verified
    # A human top-up keeps it verified ...
    more = _grades(tmp, {"q01": {"remoteok:3": 0}}, "more.json")
    module.main([more, "--grader", "human", "--note", "by me"])
    assert golden.load()[0].verified
    # ... and one LLM grade on it means not every judgement is a human's.
    llm = _grades(tmp, {"q01": {"remoteok:4": 1}}, "llm.json")
    module.main([llm, "--grader", "llm", "--note", "by a model"])
    assert not golden.load()[0].verified


def test_a_fresh_query_graded_by_a_human_is_verified(apply):
    module, tmp = apply
    golden.save(golden.load() + [golden.GoldenQuery(id="q02", query="new")])
    grades = _grades(tmp, {"q02": {"greenhouse:a:1": 2}})
    module.main([grades, "--grader", "human", "--note", "by me"])
    assert {q.id: q.verified for q in golden.load()} == {"q01": False, "q02": True}


def test_the_grader_flag_is_required(apply):
    module, tmp = apply
    with pytest.raises(SystemExit):
        module.main([_grades(tmp, {"q01": {"remoteok:2": 2}}), "--note", "x"])
