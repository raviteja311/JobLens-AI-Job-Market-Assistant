"""Scoring /chat against the golden QA set, and A/B-ing prompt versions.

`refusal_rate` is reported separately from the judge scores and is the metric
to watch. The questions marked must_refuse are the ones the corpus cannot
answer, and a system that answers them anyway is worse than one that answers
nothing: it is confidently wrong in a way the user cannot detect. That number
is a hard gate in CI, not a trend line.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from joblens import db
from joblens.eval import judge as judging
from joblens.rag import chat as chat_rag
from joblens.search.embeddings import get_embedder

CHAT_FILE = Path(__file__).resolve().parents[3] / "data" / "golden" / "chat.yaml"


@dataclass
class ChatCase:
    id: str
    question: str
    expected_facts: list[str] = field(default_factory=list)
    must_refuse: bool = False


def load_cases(path: Path | None = None) -> list[ChatCase]:
    path = path or CHAT_FILE
    if not path.exists():
        return []
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [
        ChatCase(
            id=entry["id"],
            question=entry["question"],
            expected_facts=entry.get("expected_facts", []) or [],
            must_refuse=bool(entry.get("must_refuse", False)),
        )
        for entry in payload.get("questions", [])
    ]


@dataclass
class CaseResult:
    case: ChatCase
    answer: str
    sources: int
    cited: int
    faithfulness: int
    completeness: int
    refused: bool
    correct_refusal: bool
    took_ms: int
    cost_usd: float
    unsupported: list[str] = field(default_factory=list)


@dataclass
class GenerationReport:
    prompt_version: str
    results: list[CaseResult]
    seconds: float

    @property
    def scores(self) -> dict[str, float]:
        if not self.results:
            return {}
        n = len(self.results)
        refusal_cases = [r for r in self.results if r.case.must_refuse]
        return {
            "faithfulness": sum(r.faithfulness for r in self.results) / n / 2,
            "completeness": sum(r.completeness for r in self.results) / n / 2,
            "citation_rate": sum(1 for r in self.results if r.cited) / n,
            "refusal_accuracy": (
                sum(1 for r in refusal_cases if r.correct_refusal) / len(refusal_cases)
                if refusal_cases
                else 1.0
            ),
            "cost_usd": sum(r.cost_usd for r in self.results),
        }

    def as_table(self) -> str:
        s = self.scores
        return "\n".join(
            [
                f"prompt chat_answer/{self.prompt_version}, "
                f"{len(self.results)} questions, {self.seconds:.0f}s",
                "",
                "| metric | score |",
                "| --- | ---: |",
                f"| faithfulness | {s.get('faithfulness', 0):.2f} |",
                f"| completeness | {s.get('completeness', 0):.2f} |",
                f"| citation rate | {s.get('citation_rate', 0):.2f} |",
                f"| refusal accuracy | {s.get('refusal_accuracy', 0):.2f} |",
                f"| cost | ${s.get('cost_usd', 0):.4f} |",
            ]
        )


# Refusal is detected structurally, by whether the answer cited a source,
# not by matching phrases.
#
# The first version was a list of phrases and it was wrong three times in a
# row on answers that were correct. "None of the postings mention Elon Musk
# as a founder" is a refusal. So is "I'm not able to answer that question.
# The job postings don't mention the capital of Peru." Each miss was fixed by
# adding another phrase, which is tuning the metric until it agrees with the
# answers in front of you: the measurement then reports whatever it was most
# recently taught to report.
#
# A refusal has no citations, because there is nothing to cite. A real answer
# has at least one, because the prompt requires one per claim. That holds
# regardless of how the model words it, and it was right on all 8 questions
# across both prompt versions where the phrase list scored 3 of them wrong.
#
# The failure mode it cannot see is a real answer that forgot to cite, which
# would read as a refusal. That is what `citation_rate` is for, and the two
# metrics are reported side by side so a swap between them is visible.


def is_refusal(cited: set[int] | list[int], sources: int) -> bool:
    """True when the system declined rather than answered."""
    return sources == 0 or not cited


def run(
    cases: list[ChatCase] | None = None,
    prompt_version: str | None = None,
    judge_enabled: bool = True,
) -> GenerationReport:
    cases = cases if cases is not None else load_cases()
    if not cases:
        raise ValueError(f"no chat cases in {CHAT_FILE}")

    embedder = get_embedder()
    embedder.encode(["warm up"])
    began = time.perf_counter()
    results: list[CaseResult] = []

    with db.connect() as conn:
        for case in cases:
            answer = chat_rag.ask(
                conn,
                case.question,
                embedder=embedder,
                limit=6,
                prompt_version=prompt_version,
            )
            refused = is_refusal(answer.cited, len(answer.sources))
            cost = answer.cost_usd

            faithfulness = completeness = 0
            unsupported: list[str] = []
            if judge_enabled:
                rendered = "\n\n".join(s.render() for s in answer.sources)
                verdict, judge_cost, _ = judging.judge(
                    case.question, answer.answer, rendered
                )
                faithfulness = verdict.faithfulness
                completeness = verdict.completeness
                unsupported = verdict.unsupported_claims
                cost += judge_cost

            results.append(
                CaseResult(
                    case=case,
                    answer=answer.answer,
                    sources=len(answer.sources),
                    cited=len(answer.cited),
                    faithfulness=faithfulness,
                    completeness=completeness,
                    refused=refused,
                    correct_refusal=(refused == case.must_refuse),
                    took_ms=answer.took_ms,
                    cost_usd=cost,
                    unsupported=unsupported,
                )
            )

    return GenerationReport(
        prompt_version=prompt_version or "latest",
        results=results,
        seconds=time.perf_counter() - began,
    )


def ab_test(version_a: str, version_b: str, judge_enabled: bool = True) -> str:
    """Same questions, two prompt versions, scores side by side.

    The point of having prompts in versioned files: this function is a loop
    over two directory entries rather than a git stash.
    """
    cases = load_cases()
    a = run(cases, prompt_version=version_a, judge_enabled=judge_enabled)
    b = run(cases, prompt_version=version_b, judge_enabled=judge_enabled)
    keys = ["faithfulness", "completeness", "citation_rate", "refusal_accuracy"]
    lines = [
        f"A/B over {len(cases)} questions",
        "",
        f"| metric | {version_a} | {version_b} | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in keys:
        left, right = a.scores.get(key, 0.0), b.scores.get(key, 0.0)
        lines.append(f"| {key} | {left:.2f} | {right:.2f} | {right - left:+.2f} |")
    return "\n".join(lines)
