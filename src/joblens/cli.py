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
import json
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


def cmd_skills(args: argparse.Namespace) -> int:
    """Phase 2 baseline skill extraction over what is in the database."""
    from joblens.ml import dataset, skills

    frame = dataset.load_postings(source=args.source)
    counts = skills.skill_counts(frame)
    for skill, count in counts.head(args.top).items():
        print(f"  {skill:<24} {count:>5}  {count / len(frame):>6.1%}")
    return 0


def cmd_train_salary(args: argparse.Namespace) -> int:
    from joblens.ml import dataset, salary_model

    frame = salary_model.training_frame(dataset.load_postings())
    report = salary_model.compare_models(frame, folds=args.folds)
    print(report.as_table())
    if args.save:
        path = salary_model.fit_and_save(frame, report.best.name)
        log.info("saved %s model to %s", report.best.name, path)
    return 0


def cmd_cluster(args: argparse.Namespace) -> int:
    from joblens.ml import clustering, dataset

    frame = dataset.load_postings()
    result = clustering.cluster_postings(frame, k=args.k)
    print(result.as_table())
    return 0


def cmd_trends(args: argparse.Namespace) -> int:
    from joblens.ml import dataset, trends

    frame = dataset.load_postings()
    summary = trends.summary(frame, days=args.days)
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    """Phase 3: build the vector index."""
    from joblens.search.embeddings import get_embedder
    from joblens.search.index import build_index, index_stats

    embedder = get_embedder(args.embedder)
    for strategy in args.strategy or ["whole"]:
        log.info(build_index(strategy, embedder, rebuild=args.rebuild).summary())
    for row in index_stats():
        print(
            f"  {row['strategy']:<8} {row['model']:<20} {row['chunks']:>6} chunks"
            f"  {row['postings']:>5} postings  avg {row['avg_chars']} chars"
        )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from joblens.search import retrieval

    with db.connect() as conn:
        hits = retrieval.search(
            conn,
            args.query,
            mode=args.mode,
            strategy=args.strategy,
            limit=args.limit,
            rerank=args.rerank,
        )
    for i, hit in enumerate(hits, start=1):
        where = "remote" if hit.is_remote else (hit.location or "")
        print(f"{i:>3}. {hit.score:7.4f}  {hit.title[:60]}")
        print(f"      {hit.company[:40]:<40} {where[:28]:<28} {hit.ranks}")
    return 0


def cmd_dedup(args: argparse.Namespace) -> int:
    from joblens.search import dedup

    print(dedup.compare_to_hash(args.threshold))
    if args.save:
        log.info("recorded %s pairs", dedup.record(dedup.candidates(args.threshold)))
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from joblens.rag import chat as chat_rag

    with db.connect() as conn:
        answer = chat_rag.ask(conn, args.question, limit=args.limit)
    print(answer.answer)
    print()
    for source in answer.sources:
        print(f"  [{source.n}] {source.title[:56]} | {source.company[:26]}")
        print(f"       {source.url}")
    print(f"\ngrounded={answer.grounded}  {answer.took_ms}ms  ${answer.cost_usd:.5f}")
    return 0


def cmd_match(args: argparse.Namespace) -> int:
    from pathlib import Path as _Path

    from joblens.rag import resume as resume_rag

    data = _Path(args.resume).read_bytes()
    text = (
        resume_rag.pdf_to_text(data)
        if args.resume.lower().endswith(".pdf")
        else data.decode("utf-8", errors="replace")
    )
    with db.connect() as conn:
        report = resume_rag.match_resume(conn, text, limit=args.limit)
    print(f"LLM skills : {', '.join(report.profile.skills)}")
    print(f"rule-based : {', '.join(report.rule_based_skills)}")
    print()
    for match in report.matches:
        verdict = match.verdict
        print(f"  {verdict.fit_score:>3}  {match.title[:52]} | {match.company[:24]}")
        print(f"       {verdict.reasoning[:150]}")
        print(f"       missing: {', '.join(verdict.missing_skills) or 'nothing'}")
    print(f"\n{report.took_ms}ms  ${report.cost_usd:.5f}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Phase 5: the scorecard, and the gate that fails CI."""
    from joblens.eval import report as reporting

    failures = []
    if args.suite in ("retrieval", "all"):
        from joblens.eval import retrieval as eval_retrieval

        result = eval_retrieval.run()
        print(result.as_table())
        print()
        best = result.best
        reporting.record("retrieval", best.name, result.queries, best.scores)
        gate = reporting.gate("retrieval", best.name, best.scores)
        print(gate.summary())
        failures.extend(gate.failures)

    if args.suite in ("chat", "all"):
        from joblens.eval import generation

        result = generation.run(judge_enabled=not args.no_judge)
        print()
        print(result.as_table())
        print()
        scores = result.scores
        reporting.record("chat", result.prompt_version, len(result.results), scores)
        gate = reporting.gate("chat", result.prompt_version, scores)
        print(gate.summary())
        failures.extend(gate.failures)

    return 1 if (failures and args.strict) else 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from joblens.eval import judge as judging

    calibration = judging.calibrate()
    if calibration.n == 0:
        print(
            "no hand-scored answers in data/golden/judgements.yaml. Score 20 "
            "answers by hand first: an uncalibrated judge is a number generator."
        )
        return 1
    print(calibration.as_table())
    for row in calibration.disagreements[:10]:
        print(f"\n  human {row['human']} judge {row['judge']}: {row['question']}")
        print(f"    {row['judge_reasoning'][:160]}")
    return 0


def cmd_ab(args: argparse.Namespace) -> int:
    from joblens.eval import generation

    print(generation.ab_test(args.a, args.b, judge_enabled=not args.no_judge))
    return 0


def cmd_spend(args: argparse.Namespace) -> int:
    from joblens.llm.client import spend

    for row in spend(args.days):
        print(
            f"  {row['day']}  {row['feature']:<14} {row['backend']:<10}"
            f" {row['calls']:>4} calls  {row['tokens'] or 0:>7} tokens"
            f"  ${row['usd']:>8}  {row['avg_ms'] or 0:>6}ms"
            f"  retries {row['avg_attempts']}  failed {row['failures']}"
        )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "joblens.api.main:app", host=args.host, port=args.port, reload=args.reload
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

    skills_cmd = sub.add_parser("skills", help="top skills in the corpus")
    skills_cmd.add_argument("--source", choices=sorted(sources.REGISTRY))
    skills_cmd.add_argument("--top", type=int, default=25)
    skills_cmd.set_defaults(func=cmd_skills)

    train = sub.add_parser("train-salary", help="compare salary regression models")
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--save", action="store_true", help="persist the best model")
    train.set_defaults(func=cmd_train_salary)

    cluster = sub.add_parser("cluster", help="cluster postings and label the clusters")
    cluster.add_argument(
        "--k", type=int, default=None, help="default: pick by silhouette"
    )
    cluster.set_defaults(func=cmd_cluster)

    trends_cmd = sub.add_parser("trends", help="skill and demand trends")
    trends_cmd.add_argument("--days", type=int, default=90)
    trends_cmd.set_defaults(func=cmd_trends)

    embed = sub.add_parser("embed", help="build the vector index")
    embed.add_argument("--strategy", action="append", choices=["whole", "section"])
    embed.add_argument("--embedder", default="local", choices=["local", "api"])
    embed.add_argument("--rebuild", action="store_true")
    embed.set_defaults(func=cmd_embed)

    search_cmd = sub.add_parser("search", help="search the postings")
    search_cmd.add_argument("query")
    search_cmd.add_argument(
        "--mode", default="hybrid", choices=["keyword", "vector", "hybrid"]
    )
    search_cmd.add_argument("--strategy", default="whole", choices=["whole", "section"])
    search_cmd.add_argument("--limit", type=int, default=10)
    search_cmd.add_argument("--rerank", action="store_true")
    search_cmd.set_defaults(func=cmd_search)

    dedup_cmd = sub.add_parser("dedup", help="embedding dedup vs the hash baseline")
    dedup_cmd.add_argument("--threshold", type=float, default=0.93)
    dedup_cmd.add_argument("--save", action="store_true")
    dedup_cmd.set_defaults(func=cmd_dedup)

    chat_cmd = sub.add_parser("chat", help="ask a grounded question")
    chat_cmd.add_argument("question")
    chat_cmd.add_argument("--limit", type=int, default=8)
    chat_cmd.set_defaults(func=cmd_chat)

    match_cmd = sub.add_parser("match", help="rank jobs against a resume")
    match_cmd.add_argument("resume", help="path to a PDF or text resume")
    match_cmd.add_argument("--limit", type=int, default=5)
    match_cmd.set_defaults(func=cmd_match)

    eval_cmd = sub.add_parser("eval", help="score retrieval and chat")
    eval_cmd.add_argument(
        "--suite", default="all", choices=["all", "retrieval", "chat"]
    )
    eval_cmd.add_argument("--no-judge", action="store_true")
    eval_cmd.add_argument(
        "--strict", action="store_true", help="exit 1 on a gate failure"
    )
    eval_cmd.set_defaults(func=cmd_eval)

    calibrate = sub.add_parser("calibrate", help="score the judge against human labels")
    calibrate.set_defaults(func=cmd_calibrate)

    ab = sub.add_parser("ab", help="compare two prompt versions")
    ab.add_argument("a")
    ab.add_argument("b")
    ab.add_argument("--no-judge", action="store_true")
    ab.set_defaults(func=cmd_ab)

    spend_cmd = sub.add_parser("spend", help="what the LLM calls have cost")
    spend_cmd.add_argument("--days", type=int, default=7)
    spend_cmd.set_defaults(func=cmd_spend)

    serve = sub.add_parser("serve", help="run the API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
