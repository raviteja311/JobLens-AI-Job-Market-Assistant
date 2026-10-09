"""How far two sets of relevance grades agree.

    python scripts/golden_agreement.py first.json second.json
    python scripts/golden_agreement.py data/golden/retrieval.yaml llm.json --show 10

Each side is a grades JSON (query id -> {"source:source_id": grade}, the
format scripts/golden_apply.py reads) or a golden YAML in the
data/golden/retrieval.yaml schema. Only (query, posting) pairs graded on both
sides are compared, so a subset can be checked against the full set.

Two uses, both in docs/labelling-guide.md: a human's blind re-grade against
their own first pass (self-agreement), and the LLM judge against the human
(whether LLM labels can be trusted, measured rather than assumed).

Kappa is linearly weighted because grades are ordinal: a 2 graded as 1 is a
smaller disagreement than a 2 graded as 0. Plain agreement is printed too,
but it flatters a skewed set; most candidates are 0s, and two graders who
both say 0 to everything agree 90% of the time with a kappa of zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, confusion_matrix

from joblens.eval import golden

GRADES = [0, 1, 2]


def load_grades(path: Path) -> dict[str, dict[str, int]]:
    """Grades from a JSON file or a golden YAML. Ungraded (null) entries in
    a half-filled template are left out rather than read as zeros."""
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            qid: {key: int(g) for key, g in graded.items() if g is not None}
            for qid, graded in payload.items()
        }
    return {q.id: dict(q.judgements) for q in golden.load(path)}


def paired(
    a: dict[str, dict[str, int]], b: dict[str, dict[str, int]]
) -> list[tuple[str, str, int, int]]:
    """(query, posting, grade in a, grade in b) for every pair graded in both."""
    rows = []
    for qid in sorted(set(a) & set(b)):
        for key in sorted(set(a[qid]) & set(b[qid])):
            rows.append((qid, key, a[qid][key], b[qid][key]))
    return rows


def agreement(rows: list[tuple[str, str, int, int]]) -> dict:
    left = [r[2] for r in rows]
    right = [r[3] for r in rows]
    return {
        "pairs": len(rows),
        "queries": len({r[0] for r in rows}),
        "exact": sum(x == y for x, y in zip(left, right, strict=True)) / len(rows),
        "kappa_linear": cohen_kappa_score(left, right, labels=GRADES, weights="linear"),
        "kappa": cohen_kappa_score(left, right, labels=GRADES),
        "confusion": confusion_matrix(left, right, labels=GRADES).tolist(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--show", type=int, default=0, help="list N disagreements")
    args = parser.parse_args(argv)

    rows = paired(load_grades(args.a), load_grades(args.b))
    if not rows:
        print("no (query, posting) pair is graded in both")
        return 1
    result = agreement(rows)
    print(
        f"{result['pairs']} pairs over {result['queries']} queries\n"
        f"exact agreement      {result['exact']:.3f}\n"
        f"kappa, linear weight {result['kappa_linear']:.3f}\n"
        f"kappa, unweighted    {result['kappa']:.3f}\n"
        f"\nrows: {args.a.name}, columns: {args.b.name}\n"
        "         0     1     2"
    )
    for grade, counts in zip(GRADES, result["confusion"], strict=True):
        print(f"   {grade} " + "".join(f"{n:6d}" for n in counts))
    for qid, key, x, y in [r for r in rows if r[2] != r[3]][: args.show]:
        print(f"  {qid} {key}: {x} vs {y}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
