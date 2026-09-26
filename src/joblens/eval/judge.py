"""LLM-as-judge, and the calibration that makes it worth anything.

A judge is a model grading a model, and the failure mode is that it agrees
with itself. Left uncalibrated it will hand out 2s for fluent answers and
nobody will notice, because the only thing checking the judge is the judge.

So the judge is scored against hand-scored answers before its numbers are
used anywhere. `calibrate()` reports exact agreement and Cohen's kappa
against the human labels in data/golden/judgements.yaml. Kappa rather than
raw agreement because these grades are heavily skewed towards 2: a judge that
answers "2" to everything scores about 70% agreement and a kappa of zero,
and only one of those two numbers tells you it is useless.

Known biases this design does not fix, only limits:
  position bias   not applicable, one answer is graded at a time
  verbosity bias  real and unmitigated; a longer answer tends to score higher
  self-preference the judge is the same model that wrote the answer, which is
                  the worst case. Calibration is what makes it visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from joblens.llm import client, prompts

GOLDEN_DIR = Path(__file__).resolve().parents[3] / "data" / "golden"
JUDGEMENTS_FILE = GOLDEN_DIR / "judgements.yaml"


class Verdict(BaseModel):
    faithfulness: int = Field(ge=0, le=2)
    completeness: int = Field(ge=0, le=2)
    unsupported_claims: list[str] = Field(default_factory=list)
    reasoning: str = ""


@dataclass
class JudgedAnswer:
    question_id: str
    question: str
    verdict: Verdict
    cost_usd: float = 0.0
    attempts: int = 1


def judge(
    question: str,
    answer: str,
    sources: str,
    prompt_version: str | None = None,
) -> tuple[Verdict, float, int]:
    prompt = prompts.load("judge_answer", prompt_version)
    result = client.complete_structured(
        prompt.render(question=question, answer=answer, sources=sources),
        Verdict,
        feature="judge",
        prompt=prompt,
        max_tokens=600,
    )
    return result.value, result.completion.cost_usd, result.attempts


def _kappa(human: list[int], machine: list[int], categories=(0, 1, 2)) -> float:
    """Cohen's kappa: agreement above what guessing the base rate would give.

    1.0 is perfect, 0.0 is chance, below 0 is worse than chance. Anything
    under about 0.4 means the judge's numbers should not be quoted.
    """
    n = len(human)
    if n == 0:
        return 0.0
    observed = sum(1 for h, m in zip(human, machine, strict=True) if h == m) / n
    expected = sum((human.count(c) / n) * (machine.count(c) / n) for c in categories)
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


@dataclass
class Calibration:
    n: int
    faithfulness_agreement: float
    faithfulness_kappa: float
    completeness_agreement: float
    completeness_kappa: float
    disagreements: list[dict] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        """Whether the judge's numbers are worth reporting at all."""
        return self.n >= 20 and self.faithfulness_kappa >= 0.4

    def as_table(self) -> str:
        verdict = "usable" if self.trustworthy else "NOT usable"
        return "\n".join(
            [
                f"calibrated against {self.n} hand-scored answers: {verdict}",
                "",
                "| dimension | agreement | kappa |",
                "| --- | ---: | ---: |",
                f"| faithfulness | {self.faithfulness_agreement:.1%} "
                f"| {self.faithfulness_kappa:.2f} |",
                f"| completeness | {self.completeness_agreement:.1%} "
                f"| {self.completeness_kappa:.2f} |",
            ]
        )


def _has_scores(entry: dict) -> bool:
    return all(
        isinstance(entry.get(k), int) and not isinstance(entry.get(k), bool)
        for k in ("faithfulness", "completeness")
    )


def load_human_scores(path: Path | None = None) -> list[dict]:
    """Entries a reviewer has finished scoring.

    `scripts/calibration_draft.py` writes the file with both scores empty, and
    a reviewer fills them in over time. An entry with either score missing is
    left out rather than read as a zero: half-scored is not scored, and
    running the judge on it would spend a call to compare against nothing.
    """
    path = path or JUDGEMENTS_FILE
    if not path.exists():
        return []
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [e for e in payload.get("scored", []) or [] if _has_scores(e)]


def pending_count(path: Path | None = None) -> int:
    """Drafted answers still waiting for a reviewer's scores."""
    path = path or JUDGEMENTS_FILE
    if not path.exists():
        return 0
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return sum(1 for e in payload.get("scored", []) or [] if not _has_scores(e))


def calibrate(scored: list[dict] | None = None) -> Calibration:
    """Run the judge over answers a human already graded and compare.

    Every entry needs question, answer, sources, and the human's two scores.
    """
    scored = scored if scored is not None else load_human_scores()
    if not scored:
        return Calibration(0, 0.0, 0.0, 0.0, 0.0)

    human_f, human_c, machine_f, machine_c = [], [], [], []
    disagreements = []
    for entry in scored:
        verdict, _, _ = judge(
            entry["question"], entry["answer"], entry.get("sources", "")
        )
        human_f.append(int(entry["faithfulness"]))
        human_c.append(int(entry["completeness"]))
        machine_f.append(verdict.faithfulness)
        machine_c.append(verdict.completeness)
        if verdict.faithfulness != int(entry["faithfulness"]):
            disagreements.append(
                {
                    "question": entry["question"],
                    "human": int(entry["faithfulness"]),
                    "judge": verdict.faithfulness,
                    "judge_reasoning": verdict.reasoning,
                }
            )

    n = len(scored)
    return Calibration(
        n=n,
        faithfulness_agreement=sum(
            1 for h, m in zip(human_f, machine_f, strict=True) if h == m
        )
        / n,
        faithfulness_kappa=_kappa(human_f, machine_f),
        completeness_agreement=sum(
            1 for h, m in zip(human_c, machine_c, strict=True) if h == m
        )
        / n,
        completeness_kappa=_kappa(human_c, machine_c),
        disagreements=disagreements,
    )
