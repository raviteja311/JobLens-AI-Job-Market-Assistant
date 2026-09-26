"""Dump the labelling pool for golden queries, with the full posting text.

    python scripts/golden_pool.py q01 q02 ...        selected queries
    python scripts/golden_pool.py --all               every query in the file
    python scripts/golden_pool.py --all --unjudged    only candidates never graded

For each query, every candidate that any retriever surfaces in its top 10 is
printed with its (source, source_id) key and the cleaned description, so a
grader reads the posting rather than a snippet. Grades go back into
data/golden/retrieval.yaml through `scripts/golden_apply.py`.

`--unjudged` is the top-up after a re-ingest: boards expire postings and new
ones take their place in the top 10, and an ungraded posting scores as
irrelevant, so only those need a grader's time.
"""

from __future__ import annotations

import argparse
import sys

from joblens import db
from joblens.cleaning import strip_noise
from joblens.eval import golden, label
from joblens.search.embeddings import get_embedder

MAX_CHARS = 1400


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ids", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument(
        "--unjudged", action="store_true", help="skip candidates already graded"
    )
    args = parser.parse_args(argv)

    queries = golden.load()
    if not args.all:
        wanted = set(args.ids)
        queries = [q for q in queries if q.id in wanted]
    embedder = get_embedder()
    embedder.encode(["warm up"])

    out = sys.stdout
    with db.connect() as conn:
        for query in queries:
            candidates = label.pool(conn, query.query, embedder, depth=args.depth)
            if args.unjudged:
                candidates = [c for c in candidates if c.key not in query.judgements]
                if not candidates:
                    continue
            keys = [c.posting_id for c in candidates]
            rows = conn.execute(
                "select id, description, salary_raw, posted_at from postings"
                " where id = any(%s)",
                (keys,),
            ).fetchall()
            detail = {r["id"]: r for r in rows}
            out.write(f"\n{'=' * 78}\nQUERY {query.id}: {query.query}\n")
            if query.note:
                out.write(f"NOTE: {query.note}\n")
            out.write(f"{len(candidates)} candidates\n")
            for c in candidates:
                row = detail.get(c.posting_id, {})
                where = c.location or ("remote" if c.is_remote else "not stated")
                if c.location and c.is_remote:
                    where += ", remote"
                out.write(
                    f"\n--- {c.key} | {c.title} | {c.company} | {where}"
                    f" | salary: {row.get('salary_raw') or 'not stated'}"
                    f" | found by: {', '.join(c.found_by)}\n"
                )
                body = strip_noise(row.get("description") or "")
                out.write(body[:MAX_CHARS] + ("..." if len(body) > MAX_CHARS else ""))
                out.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
