"""Merge graded judgements into data/golden/retrieval.yaml.

    python scripts/golden_apply.py grades.json [more.json ...] --note "..."

Each JSON file maps query id -> {"source:source_id": grade}. Grades are
0, 1 or 2. Zeros are kept: "judged and rejected" and "never looked at" are
different states. Every merged query is marked verified and the provenance
given on the command line is added to its note (or replaces it, with
--replace), so the file always says who graded it and from what.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from joblens.eval import golden, label


def merge_note(note: str, provenance: str, replace: bool = False) -> str:
    """The query's intent, then every grader whose grades are in the file.

    A merge adds grades on top of the old ones, so it adds its provenance on
    top of the old provenance too; overwriting it would credit a 105-posting
    top-up with the 1,274 grades underneath it. `--replace` drops the old
    grades, and with them the old provenance.
    """
    parts = [p.strip() for p in note.split(" | ") if p.strip()]
    intent = [p for p in parts[:1] if not p.startswith("judged by")]
    earlier = [] if replace else [p for p in parts if p not in intent]
    if provenance not in earlier:
        earlier.append(provenance)
    return " | ".join(intent + earlier)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--note", required=True, help="provenance for the note field")
    parser.add_argument("--replace", action="store_true", help="drop old judgements")
    args = parser.parse_args(argv)

    queries = {q.id: q for q in golden.load()}
    merged = 0
    for path in args.files:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for query_id, graded in payload.items():
            query = queries.get(query_id)
            if query is None:
                print(f"skipping unknown query {query_id}")
                continue
            bad = {k: v for k, v in graded.items() if int(v) not in (0, 1, 2)}
            if bad:
                raise SystemExit(f"{query_id}: grades must be 0, 1 or 2: {bad}")
            if args.replace:
                query.judgements = {}
            label.apply_judgements(query, graded, verified=True)
            query.note = merge_note(query.note, args.note, replace=args.replace)
            merged += 1
    path = golden.save(list(queries.values()))
    verified = [q for q in queries.values() if q.verified and q.relevant_keys]
    print(
        f"merged {merged} queries into {path}; {len(verified)} verified with relevant"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
