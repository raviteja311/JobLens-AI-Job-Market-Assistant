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
    assert module.main([grades, "--note", "judged by a reviewer"]) == 0
    (query,) = golden.load()
    assert query.judgements == {"remoteok:1": 2, "hackernews:9": 0, "remoteok:2": 1}
    assert query.verified
    assert query.note == (
        "rare-token case | judged by an older pass | judged by a reviewer"
    )
    # Applying the same batch twice does not stack the note.
    module.main([grades, "--note", "judged by a reviewer"])
    assert golden.load()[0].note.count("judged by a reviewer") == 1


def test_replace_drops_old_judgements(apply):
    module, tmp = apply
    grades = _grades(tmp, {"q01": {"remoteok:2": 2}})
    module.main([grades, "--note", "judged by a reviewer", "--replace"])
    (query,) = golden.load()
    assert query.judgements == {"remoteok:2": 2}
    assert query.note == "rare-token case | judged by a reviewer"


def test_out_of_range_grade_is_refused_before_anything_is_written(apply):
    module, tmp = apply
    grades = _grades(tmp, {"q01": {"remoteok:2": 3}})
    with pytest.raises(SystemExit, match="0, 1 or 2"):
        module.main([grades, "--note", "x"])
    assert golden.load()[0].judgements == {"remoteok:1": 2}


def test_unknown_query_ids_are_skipped(apply, capsys):
    module, tmp = apply
    module.main([_grades(tmp, {"q99": {"remoteok:2": 1}}), "--note", "x"])
    assert "skipping unknown query q99" in capsys.readouterr().out
