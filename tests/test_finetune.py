"""The distillation plumbing, tested without loading a model.

None of what is checked here is about whether a 0.5B model is any good. It is
about whether the dataset format, the scoring and the guard against grading a
student on its own teacher's labels do what they claim.
"""

import json

import pytest

from joblens.finetune import dataset, evaluate
from joblens.finetune.extract import Usage, _normalise


def example(
    posting_id: int = 1,
    teacher: list[str] | None = None,
    rule_based: list[str] | None = None,
    label: list[str] | None = None,
    verified: bool = True,
) -> dataset.Example:
    teacher = ["python", "pytorch"] if teacher is None else teacher
    rule_based = ["python"] if rule_based is None else rule_based
    return dataset.Example(
        posting_id=posting_id,
        source="hackernews",
        source_id=str(posting_id),
        title="ML Engineer",
        description="We use Python and PyTorch.",
        teacher=teacher,
        rule_based=rule_based,
        label=teacher if label is None else label,
        verified=verified,
    )


class StubExtractor:
    """Returns whatever it was told to, so scoring can be checked exactly."""

    def __init__(self, name, answers):
        self.name = name
        self.answers = answers
        self.usage = Usage()

    def extract(self, title, description):
        return self.answers.pop(0)


def test_render_strips_boilerplate_and_truncates():
    text = dataset.render("ML Engineer", "Apply at https://x.com . " + "word " * 2000)
    assert "https://x.com" not in text
    assert text.startswith("Title: ML Engineer")
    assert len(text) < dataset.MAX_DESCRIPTION_CHARS + 200


def test_example_key_is_the_source_id_not_the_primary_key():
    # The same reason the golden set does it: posting.id is renumbered by a
    # re-ingest, the source's own id is not.
    assert example(7).key == "hackernews:7"


def test_disagreement_is_the_symmetric_difference():
    row = example(teacher=["python", "pytorch"], rule_based=["python", "aws"])
    assert row.disagreement == {"pytorch", "aws"}


def test_to_chat_has_a_user_turn_and_a_json_answer():
    row = dataset.to_chat(example())
    assert [m["role"] for m in row["messages"]] == ["user", "assistant"]
    assert "ML Engineer" in row["messages"][0]["content"]
    payload = json.loads(row["messages"][1]["content"])
    assert payload == {"skills": ["python", "pytorch"]}


def test_to_chat_sorts_the_answer():
    # Ordering is signal the model cannot learn and loss it cannot reduce.
    row = dataset.to_chat(example(teacher=["pytorch", "aws", "python"]))
    assert json.loads(row["messages"][1]["content"])["skills"] == [
        "aws",
        "python",
        "pytorch",
    ]


def test_split_is_deterministic_and_disjoint():
    rows = [example(i) for i in range(20)]
    train, test = dataset.split(rows, test_size=5, seed=1)
    again, _ = dataset.split(rows, test_size=5, seed=1)
    assert len(test) == 5 and len(train) == 15
    assert {e.posting_id for e in train} & {e.posting_id for e in test} == set()
    assert [e.posting_id for e in train] == [e.posting_id for e in again]


def test_round_trip_through_jsonl(tmp_path):
    rows = [example(1), example(2)]
    path = dataset.save(rows, tmp_path / "x.jsonl")
    loaded = dataset.load(path)
    assert [e.posting_id for e in loaded] == [1, 2]
    assert loaded[0].label == ["python", "pytorch"]


def test_stats_counts_where_the_two_labellers_disagree():
    rows = [
        example(1, teacher=["python", "pytorch"], rule_based=["python"]),
        example(2, teacher=["aws"], rule_based=["aws", "docker"]),
    ]
    stats = dataset.stats(rows)
    assert stats["examples"] == 2
    assert stats["teacher_only_mentions"] == 1
    assert stats["rule_only_mentions"] == 1


def test_stats_flags_skills_outside_the_taxonomy():
    # The teacher inventing "trigger script development" and the taxonomy
    # being out of date look identical from a count, which is why it is
    # surfaced rather than silently dropped.
    rows = [example(1, teacher=["python", "gui development"])]
    stats = dataset.stats(rows)
    assert stats["off_taxonomy_skills"] == 1
    assert "gui development" in stats["off_taxonomy_examples"]


def test_normalise_lowercases_dedupes_and_sorts():
    assert _normalise(["PyTorch", "python", "PYTHON", " aws "]) == [
        "aws",
        "python",
        "pytorch",
    ]


def test_f1_of_a_perfect_run_is_one():
    rows = [example(1, teacher=["python", "aws"])]
    stub = StubExtractor("perfect", [["python", "aws"]])
    score = evaluate.score_extractor(stub, rows)
    assert score.micro_f1 == 1.0
    assert score.macro_f1 == 1.0
    assert score.exact_match == 1.0


def test_f1_of_finding_nothing_is_zero():
    rows = [example(1, teacher=["python", "aws"])]
    score = evaluate.score_extractor(StubExtractor("empty", [[]]), rows)
    assert score.micro_f1 == 0.0
    assert score.empty_predictions == 1


def test_predicting_nothing_for_a_posting_with_no_skills_is_correct():
    # Sparse postings are a third of this corpus. Scoring an empty-and-right
    # answer as 0 would punish exactly the models that handle them.
    rows = [example(1, teacher=[])]
    score = evaluate.score_extractor(StubExtractor("empty", [[]]), rows)
    assert score.macro_f1 == 1.0
    assert score.exact_match == 1.0


def test_micro_and_macro_disagree_when_one_posting_is_dense():
    rows = [
        example(1, teacher=["a", "b", "c", "d", "e", "f", "g", "h"]),
        example(2, teacher=["z"]),
    ]
    # Gets the dense posting fully right and the sparse one wrong.
    stub = StubExtractor("dense", [["a", "b", "c", "d", "e", "f", "g", "h"], []])
    score = evaluate.score_extractor(stub, rows)
    assert score.micro_f1 > 0.85
    assert score.macro_f1 == pytest.approx(0.5)


def test_precision_and_recall_move_independently():
    rows = [example(1, teacher=["python", "aws"])]
    over = evaluate.score_extractor(
        StubExtractor("over", [["python", "aws", "java", "rust"]]), rows
    )
    assert over.micro_recall == 1.0
    assert over.micro_precision == 0.5


def test_scoring_refuses_a_test_set_nobody_verified():
    rows = [example(1, verified=False)]
    with pytest.raises(ValueError, match="no hand-corrected examples"):
        evaluate.run(rows)


def test_usage_reports_ms_per_call():
    usage = Usage(calls=4, seconds=2.0)
    assert usage.ms_per_call == 500.0
    assert Usage().ms_per_call == 0.0
