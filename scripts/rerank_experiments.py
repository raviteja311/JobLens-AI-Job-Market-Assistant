"""Phase 3: the reranker experiments, scored and judged in one run.

    python scripts/rerank_experiments.py
    python scripts/rerank_experiments.py --judgements grades.json --label "LLM judge"

Every configuration changes one thing from the shipped reranker (MiniLM-L6,
snippet text, a pool of 30, pure reorder), so a difference in the table has
one cause. The hybrid baseline and the shipped reranker run in the same pass,
on the same queries, with warm models.

The decision rule was fixed before any of these numbers existed
(docs/experiments.md, 2026-10-08): rerank is on by default only if its best
configuration gains at least +0.05 MRR over hybrid with p95 under 500 ms.
It is applied at the end, and only to the human-verified golden set: a run
scored against anything else (`--judgements`) is printed as provisional and
cannot decide anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from joblens.eval import golden
from joblens.eval import retrieval as eval_retrieval
from joblens.search.rerank import RerankConfig

MIN_MRR_GAIN = 0.05
MAX_P95_MS = 500.0
TINY = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
BASELINE = "hybrid (whole)"


def _rerank(name: str, **settings) -> dict:
    return {
        "name": name,
        "mode": "hybrid",
        "strategy": "whole",
        "rerank": True,
        "rerank_config": RerankConfig(**settings),
    }


CONFIGURATIONS = (
    {"name": BASELINE, "mode": "hybrid", "strategy": "whole", "rerank": False},
    _rerank("rerank: shipped (snippet, pool 30)"),
    # 3.2 what the cross-encoder reads
    _rerank("3.2 document = head", document="head"),
    _rerank("3.2 document = best section", document="section"),
    # 3.3 candidate pool
    _rerank("3.3 pool = 20", candidates=20),
    _rerank("3.3 pool = 50", candidates=50),
    # 3.4 fusion with the first stage
    _rerank("3.4 alpha = 0.3", alpha=0.3),
    _rerank("3.4 alpha = 0.5", alpha=0.5),
    _rerank("3.4 alpha = 0.7", alpha=0.7),
    # 3.5 cheaper model
    _rerank("3.5 TinyBERT-L-2", model=TINY),
    _rerank("3.5 max_length = 256", max_length=256),
    _rerank("3.5 TinyBERT-L-2, max_length = 256", model=TINY, max_length=256),
    # Combinations, added on 2026-10-08 after the provisional LLM-judged run
    # and before any human grade existed (docs/experiments.md). They pair
    # the best text (3.2) with the cheap model (3.5), so they are named here,
    # ahead of the human-judged run, rather than picked after it.
    _rerank("combo TinyBERT-L-2, best section", model=TINY, document="section"),
    _rerank(
        "combo TinyBERT-L-2, best section, max_length = 256",
        model=TINY,
        document="section",
        max_length=256,
    ),
)


def queries_from(path: Path) -> list[golden.GoldenQuery]:
    """The golden queries, with judgements taken from a grades JSON instead."""
    grades = json.loads(path.read_text(encoding="utf-8"))
    texts = {q.id: q.query for q in golden.load()}
    return [
        golden.GoldenQuery(id=qid, query=texts[qid], judgements=graded)
        for qid, graded in sorted(grades.items())
        if any(graded.values())
    ]


def verdict(report, provisional: bool) -> str:
    by_name = {c.name: c for c in report.configs}
    base = by_name[BASELINE]
    reranked = [c for c in report.configs if c.name != BASELINE]
    passing = [
        c
        for c in reranked
        if c.scores["mrr"] - base.scores["mrr"] >= MIN_MRR_GAIN
        and c.p95_ms < MAX_P95_MS
    ]
    best = max(reranked, key=lambda c: c.scores["mrr"])
    lines = [
        f"baseline {BASELINE}: MRR {base.scores['mrr']:.3f}, p95 {base.p95_ms:.0f} ms",
        f"best reranker by MRR: {best.name}: MRR {best.scores['mrr']:.3f}"
        f" ({best.scores['mrr'] - base.scores['mrr']:+.3f}), p95 {best.p95_ms:.0f} ms",
        f"configurations meeting the rule (MRR >= +{MIN_MRR_GAIN}, p95 <"
        f" {MAX_P95_MS:.0f} ms): {', '.join(c.name for c in passing) or 'none'}",
    ]
    if provisional:
        lines.append(
            "PROVISIONAL: not scored on the human-verified golden set, so this"
            " run cannot decide the default."
        )
    elif passing:
        lines.append("decision: rerank ON by default, with the passing config.")
    else:
        lines.append("decision: hybrid stays the default; rerank stays opt-in.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--judgements", type=Path, help="grades JSON (provisional)")
    parser.add_argument("--label", default="", help="who made --judgements")
    parser.add_argument("--save", type=Path, help="also write the report here")
    args = parser.parse_args(argv)

    if args.judgements:
        queries = queries_from(args.judgements)
        judged_by = f"{args.label or args.judgements.name} (PROVISIONAL)"
    else:
        queries = golden.verified_only(golden.load())
        judged_by = "human-verified golden set"
    report = eval_retrieval.run(
        queries=queries, configurations=CONFIGURATIONS, judged_by=judged_by
    )
    text = (
        report.as_table(columns=("recall@10", "mrr", "ndcg@10"))
        + "\n\n"
        + verdict(report, provisional=bool(args.judgements))
    )
    print(text)
    if args.save:
        args.save.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
