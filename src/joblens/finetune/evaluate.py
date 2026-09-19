"""Scoring the extractors against the hand-corrected test set.

Set-valued F1, not accuracy. A posting has a set of skills and an extractor
returns a set; the question is how much they overlap. Accuracy would need a
notion of "correct answer" for a task where returning four of five right
skills is obviously better than returning none.

Micro and macro are both reported because they disagree in a way that
matters. Micro pools every mention, so postings naming ten technologies
dominate it. Macro averages per posting, so a posting naming one skill counts
as much as one naming ten. A model that does well on dense engineering
postings and poorly on sparse ones scores well on micro and badly on macro,
and that gap is the thing worth seeing.

Cost per 1,000 extractions is projected from measured latency, not from a
pricing page. For the local models it is electricity, which is rounded to
zero and labelled as such rather than pretended to be free in a way that
flatters them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from joblens.finetune import dataset
from joblens.finetune.extract import Extractor, get_extractor

log = logging.getLogger(__name__)

# What an API call would have cost, for the row the comparison cannot run.
# Listed so the table has a scale, marked as an estimate everywhere it shows.
API_COST_PER_1K_EXTRACTIONS_USD = 1.50


@dataclass
class Score:
    name: str
    examples: int
    micro_f1: float
    micro_precision: float
    micro_recall: float
    macro_f1: float
    exact_match: float
    empty_predictions: int
    parse_failures: int
    ms_per_call: float
    usd: float

    @property
    def usd_per_1k(self) -> float:
        return self.usd / self.examples * 1000 if self.examples else 0.0


def _prf(predicted: set[str], truth: set[str]) -> tuple[int, int, int]:
    """Returns (true positives, predicted count, truth count)."""
    return len(predicted & truth), len(predicted), len(truth)


def _f1(precision: float, recall: float) -> float:
    return (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )


def score_extractor(extractor: Extractor, examples: list[dataset.Example]) -> Score:
    """Run one extractor over the test set and score it."""
    tp_total = pred_total = truth_total = 0
    per_example: list[float] = []
    exact = 0
    empty = 0

    began = time.perf_counter()
    for i, example in enumerate(examples, start=1):
        predicted = set(extractor.extract(example.title, example.description))
        truth = {s.lower() for s in example.label}
        if not predicted:
            empty += 1
        if predicted == truth:
            exact += 1

        tp, npred, ntruth = _prf(predicted, truth)
        tp_total += tp
        pred_total += npred
        truth_total += ntruth

        # A posting with no skills and a prediction of none is a perfect
        # answer, not a division by zero. Scoring it 0 would punish the only
        # models that get the sparse postings right.
        if not truth and not predicted:
            per_example.append(1.0)
        else:
            precision = tp / npred if npred else 0.0
            recall = tp / ntruth if ntruth else 0.0
            per_example.append(_f1(precision, recall))

        if i % 20 == 0:
            log.info("%s: scored %s of %s", extractor.name, i, len(examples))
    elapsed = time.perf_counter() - began

    micro_p = tp_total / pred_total if pred_total else 0.0
    micro_r = tp_total / truth_total if truth_total else 0.0
    return Score(
        name=extractor.name,
        examples=len(examples),
        micro_f1=_f1(micro_p, micro_r),
        micro_precision=micro_p,
        micro_recall=micro_r,
        macro_f1=sum(per_example) / len(per_example) if per_example else 0.0,
        exact_match=exact / len(examples) if examples else 0.0,
        empty_predictions=empty,
        parse_failures=extractor.usage.failures,
        ms_per_call=(elapsed * 1000 / len(examples)) if examples else 0.0,
        usd=extractor.usage.usd,
    )


@dataclass
class Comparison:
    scores: list[Score] = field(default_factory=list)

    def as_table(self) -> str:
        head = self.scores[0] if self.scores else None
        lines = [
            f"{head.examples if head else 0} hand-corrected postings",
            "",
            "| extractor | micro F1 | macro F1 | precision | recall | exact | "
            "unparseable | ms/call | $/1k |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for score in self.scores:
            lines.append(
                f"| {score.name} | {score.micro_f1:.3f} | {score.macro_f1:.3f} "
                f"| {score.micro_precision:.3f} | {score.micro_recall:.3f} "
                f"| {score.exact_match:.3f} | {score.parse_failures} "
                f"| {score.ms_per_call:.0f} | ${score.usd_per_1k:.2f} |"
            )
        return "\n".join(lines)


def run(
    examples: list[dataset.Example] | None = None,
    names: tuple[str, ...] = ("rules", "base", "tuned", "teacher"),
) -> Comparison:
    """Score every extractor on the same hand-corrected postings."""
    examples = examples if examples is not None else dataset.load(dataset.TEST_FILE)
    verified = [e for e in examples if e.verified]
    if not verified:
        raise ValueError(
            "the test set has no hand-corrected examples. Scoring a distilled "
            "student against its own teacher's labels measures imitation, not "
            "extraction, and the teacher wins by construction."
        )

    comparison = Comparison()
    for name in names:
        try:
            extractor = get_extractor(name)
        except FileNotFoundError as exc:
            log.warning("skipping %s: %s", name, exc)
            continue
        log.info("scoring %s over %s postings", name, len(verified))
        comparison.scores.append(score_extractor(extractor, verified))
    return comparison
