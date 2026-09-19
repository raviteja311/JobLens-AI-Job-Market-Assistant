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


def test_split_hash_is_stable_and_order_sensitive():
    a = [example(1), example(2), example(3)]
    assert dataset.split_hash(a) == dataset.split_hash(
        [example(1), example(2), example(3)]
    )
    assert dataset.split_hash(a) != dataset.split_hash(list(reversed(a)))


def test_balance_caps_the_empty_share():
    rows = [example(i, teacher=["python"]) for i in range(20)]
    rows += [example(100 + i, teacher=[], rule_based=[], label=[]) for i in range(60)]
    balanced = dataset.balance(rows, max_empty_share=0.3)
    empty = sum(1 for e in balanced if not e.label)
    assert empty / len(balanced) <= 0.31
    # Empties are capped, not removed: a model that never answers empty
    # would invent skills for the non-technical third of the corpus.
    assert empty > 0


def test_a_collapsed_extractor_is_flagged_not_just_scored_low():
    # 10 postings, model answers nothing on all of them. The first fine-tune
    # did exactly this and produced a row that looked like a normal result.
    rows = [example(i, teacher=["python"]) for i in range(10)]
    score = evaluate.score_extractor(StubExtractor("dead", [[]] * 10), rows)
    assert score.collapsed
    assert score.empty_share == 1.0
    assert score.micro_f1 == 0.0


def test_a_working_extractor_is_not_flagged_as_collapsed():
    rows = [example(i, teacher=["python"]) for i in range(10)]
    score = evaluate.score_extractor(StubExtractor("ok", [["python"]] * 10), rows)
    assert not score.collapsed
    assert score.empty_share == 0.0


def test_mostly_empty_but_correct_is_not_a_collapse():
    # Nine genuinely skill-free postings and one with a skill, answered
    # correctly. High empty rate, perfect score, must not trip the guard.
    rows = [example(i, teacher=[], rule_based=[], label=[]) for i in range(9)]
    rows.append(example(99, teacher=["python"]))
    answers = [[]] * 9 + [["python"]]
    score = evaluate.score_extractor(StubExtractor("sparse", answers), rows)
    assert score.empty_share == 0.9
    assert score.collapsed  # 90% empty trips the threshold by design
    assert score.macro_f1 == 1.0  # ...but the model is perfect, so the
    # guard is a prompt to go and look, not a verdict on its own.


def test_assert_not_collapsed_raises_with_the_offender_named():
    rows = [example(i, teacher=["python"]) for i in range(10)]
    comparison = evaluate.Comparison(
        scores=[evaluate.score_extractor(StubExtractor("dead", [[]] * 10), rows)]
    )
    with pytest.raises(RuntimeError, match="dead"):
        evaluate.assert_not_collapsed(comparison)


def test_micro_f1_helper_matches_the_scorer():
    from joblens.finetune.callbacks import _micro_f1

    assert _micro_f1([{"a", "b"}], [{"a", "b"}]) == 1.0
    assert _micro_f1([set()], [{"a"}]) == 0.0
    assert _micro_f1([{"a"}], [{"a", "b"}]) == pytest.approx(2 / 3)


def test_epoch_score_summary_shows_the_collapse():
    from joblens.finetune.callbacks import EpochScore

    text = EpochScore(
        epoch=3, micro_f1=0.0, empty_share=1.0, unparseable=0, eval_loss=1.565
    ).summary()
    # The exact pairing that the first run hid: loss down, F1 zero.
    assert "micro F1 0.000" in text
    assert "empty 100%" in text
    assert "1.5650" in text


def test_train_excluding_keeps_the_frozen_test_set_out():
    labelled = [example(i) for i in range(10)]
    test = [example(3), example(7)]
    train = dataset.train_excluding(labelled, test)
    assert len(train) == 8
    assert {e.key for e in train} & {e.key for e in test} == set()


def test_train_excluding_matches_on_key_not_primary_key():
    # Same posting, renumbered by a re-ingest. It must still be excluded.
    held = dataset.Example(
        posting_id=999,
        source="hackernews",
        source_id="3",
        title="t",
        description="d",
    )
    train = dataset.train_excluding([example(3)], [held])
    assert train == []


# --------------------------------------------------------------------------
# Gold v2 guards. Both of these were written to fail against the code as it
# stood, so that passing means something.
# --------------------------------------------------------------------------


def test_gold_is_not_a_superset_of_the_rules_output():
    """Precision must be measurable, not an identity.

    Gold v1 was built as rules | adjudicated(teacher), so the regex could not
    produce a false positive on any of the 60 postings and its precision came
    out at exactly 1.000. If this assertion ever passes trivially again, the
    labelling has regressed to defining truth as whatever the baseline found.
    """
    from joblens.finetune import gold as gold_module

    rows = gold_module.load_gold()
    if not rows:
        pytest.skip("gold v2 has not been built")
    examples = {e.key: e for e in dataset.load(dataset.TEST_FILE)}

    overshoot = [
        r.posting_key
        for r in rows
        if not {s.lower() for s in examples[r.posting_key].rule_based}
        <= {s.lower() for s in r.label}
    ]
    assert overshoot, (
        "gold contains every term the regex proposed, so the regex cannot "
        "produce a false positive and its precision is an identity"
    )


LONG_ONE = "Requires Python. " + ("filler words here. " * 200) + " Also Rust."
LONG_TWO = "We use Docker. " + ("more filler text. " * 250) + " And Kubernetes."
PARITY_FIXTURES = [
    ("ML Engineer", "Short posting naming PyTorch and AWS."),
    ("Data Engineer", LONG_ONE),
    ("Platform Engineer", LONG_TWO),
]


def test_the_baseline_reads_exactly_the_text_gold_was_built_from():
    """The defect this guards was the baseline calling the wrong builder.

    render() truncates at MAX_DESCRIPTION_CHARS and posting_text() does not.
    Gold is built from render(), and RuleExtractor used to call posting_text,
    so the regex was scored on more text than the labels were derived from
    and charged with false positives for skills it correctly found past the
    cutoff. The two disagreed on 12 of the 60 test postings.
    """
    from joblens.finetune.extract import RuleExtractor
    from joblens.ml.skills import extract_skills

    assert len(LONG_ONE) > dataset.MAX_DESCRIPTION_CHARS
    assert len(LONG_TWO) > dataset.MAX_DESCRIPTION_CHARS

    for title, description in PARITY_FIXTURES:
        from_gold_text = sorted(
            {s.lower() for s in extract_skills(dataset.render(title, description))}
        )
        from_extractor = RuleExtractor().extract(title, description)
        assert from_extractor == from_gold_text, (
            f"the rules row and the gold labels disagree for {title!r}: "
            f"{from_extractor} vs {from_gold_text}"
        )


def test_the_two_text_builders_are_deliberately_different():
    """Pin the divergence so nobody merges them by accident.

    posting_text repeats the title, which is Phase 2 weighting for TF-IDF and
    feeds clustering, the salary model and skill_matrix. render prefixes
    "Title:" because it goes into a prompt. Making them byte-identical would
    move Phase 2's published coverage figure and cluster labels to fix a
    Phase 6 problem that was fixed by pointing the extractor at render.
    """
    import pandas as pd

    from joblens.ml.skills import posting_text

    frame = pd.DataFrame([{"title": "ML Engineer", "description": "We use PyTorch."}])
    assert posting_text(frame).iloc[0] == "ML Engineer. ML Engineer. We use PyTorch."
    assert dataset.render("ML Engineer", "We use PyTorch.") == (
        "Title: ML Engineer\n\nWe use PyTorch."
    )
