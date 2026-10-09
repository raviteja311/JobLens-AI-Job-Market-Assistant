"""Sweep the RRF constant k for hybrid search.

    python scripts/rrf_sweep.py
    python scripts/rrf_sweep.py --judgements grades.json --label "LLM judge"

RRF scores a document as the sum of 1 / (k + rank) over the ranked lists it
appears in. k = 60 comes from Cormack, Clarke and Buettcher (2009), where it
worked well across TREC collections; nothing about this corpus chose it. A
small k trusts each list's top ranks (rank 1 is worth far more than rank 5);
a large k flattens that, so a posting both retrievers rank moderately well
overtakes one that only a single retriever ranks first.

Hybrid (whole) only. The section strategy's vector arm is truncated by the
HNSW post-filter at the default ef_search (docs/experiments.md, 2026-10-08),
which would confound a k sweep.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from joblens.eval import golden
from joblens.eval import retrieval as eval_retrieval

KS = (1, 10, 30, 60, 100, 300)


def _queries_from(path: Path) -> list[golden.GoldenQuery]:
    spec = importlib.util.spec_from_file_location(
        "rerank_experiments", Path(__file__).with_name("rerank_experiments.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.queries_from(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--judgements", type=Path, help="grades JSON (provisional)")
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)

    if args.judgements:
        queries = _queries_from(args.judgements)
        judged_by = f"{args.label or args.judgements.name} (PROVISIONAL)"
    else:
        queries = golden.verified_only(golden.load())
        judged_by = "human-verified golden set"
    configurations = [
        {
            "name": f"hybrid (whole), k = {k}" + ("  [shipped]" if k == 60 else ""),
            "mode": "hybrid",
            "strategy": "whole",
            "rerank": False,
            "rrf_k": k,
        }
        for k in KS
    ]
    report = eval_retrieval.run(
        queries=queries, configurations=configurations, judged_by=judged_by
    )
    print(report.as_table(columns=("recall@10", "mrr", "ndcg@10")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
