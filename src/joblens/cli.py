"""Command line entry point.

    python -m joblens migrate
    python -m joblens ingest --limit 200
    python -m joblens transform --source hackernews
    python -m joblens stats

One entry point instead of a folder of loose scripts. The cron job, the
GitHub Action and a human debugging a bad run all take the same code path,
so a failure in production is reproducible by typing the same command.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from joblens import db, pipeline, sources
from joblens.config import get_settings

log = logging.getLogger("joblens")


def _configure_logging(verbose: bool) -> None:
    settings = get_settings()
    level = logging.DEBUG if verbose else getattr(logging, settings.log_level, "INFO")
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_migrate(args: argparse.Namespace) -> int:
    db.migrate()
    log.info("migrations applied")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    names = args.source or sources.DEFAULT_SOURCES
    results = pipeline.run(names, limit=args.limit)
    for result in results:
        log.info(result.summary())
    # Exit non-zero if every source failed. One board being down is normal and
    # must not page anyone; all of them being down means the run is worthless.
    return 1 if results and all(r.failed for r in results) else 0


def cmd_transform(args: argparse.Namespace) -> int:
    result = pipeline.transform(args.source)
    log.info(result.summary())
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """What is actually in the database right now."""
    with db.connect() as conn:
        by_source = conn.execute("""
            select source,
                   count(*) as postings,
                   count(salary_min_year) as with_salary,
                   count(*) filter (where is_remote) as remote,
                   max(last_seen_at) as last_seen
              from postings
             group by source
             order by postings desc
            """).fetchall()
        runs = conn.execute("""
            select source, status, fetched, inserted, updated, started_at
              from ingestion_runs
             order by started_at desc
             limit 10
            """).fetchall()
        duplicates = db.count_duplicates(conn)

    total = sum(row["postings"] for row in by_source)
    print(f"postings: {total}  duplicate content hashes: {duplicates}")
    for row in by_source:
        print(
            f"  {row['source']:<12} {row['postings']:>6} postings"
            f"  {row['with_salary']:>5} with salary"
            f"  {row['remote']:>5} remote"
            f"  last seen {row['last_seen']:%Y-%m-%d %H:%M}"
        )
    print("\nlast 10 runs:")
    for row in runs:
        print(
            f"  {row['started_at']:%Y-%m-%d %H:%M}  {row['source']:<12}"
            f" {row['status']:<8} fetched {row['fetched']:>4}"
            f" new {row['inserted']:>4} refreshed {row['updated']:>4}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="joblens", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("migrate", help="create tables").set_defaults(func=cmd_migrate)

    ingest = sub.add_parser("ingest", help="fetch sources into the database")
    ingest.add_argument(
        "--source",
        action="append",
        choices=sorted(sources.REGISTRY),
        help="repeatable. Defaults to every registered source.",
    )
    ingest.add_argument("--limit", type=int, default=200)
    ingest.set_defaults(func=cmd_ingest)

    transform = sub.add_parser("transform", help="re-parse bronze into silver")
    transform.add_argument("--source", choices=sorted(sources.REGISTRY))
    transform.set_defaults(func=cmd_transform)

    sub.add_parser("stats", help="what is in the database").set_defaults(func=cmd_stats)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
