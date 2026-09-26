"""Pipeline orchestration.
Two independent halves:
    ingest_source()   network -> raw_postings   (bronze)
    transform()       raw_postings -> postings  (silver)
They are separate on purpose. When the salary parser turns out to be wrong,
`transform` re-runs over everything already collected without hitting a
single API. That is the whole reason for keeping the raw layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from joblens import db, sources
from joblens.models import Posting

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    source: str
    fetched: int = 0
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    failed: bool = False
    error: str | None = None

    def summary(self) -> str:
        if self.failed:
            return f"{self.source}: failed ({self.error})"
        return (
            f"{self.source}: fetched {self.fetched}, "
            f"new {self.inserted}, refreshed {self.updated}, skipped {self.skipped}"
        )


def _to_postings(module, rows: list[dict]) -> tuple[list[Posting], int]:
    """Parse raw rows into Postings, counting the ones we could not use.
    A parse failure on one posting must not kill the run. Job boards will
    always contain a row with a null company or an unparseable date, and
    losing 400 good postings to one bad one is not acceptable.
    """
    postings: list[Posting] = []
    skipped = 0
    for row in rows:
        try:
            posting = module.to_posting(row["payload"])
        except Exception as exc:  # noqa: BLE001 - one bad row must not stop the run
            log.debug("could not parse %s/%s: %s", row["source"], row["source_id"], exc)
            posting = None
        if posting is None:
            skipped += 1
        else:
            postings.append(posting)
    return postings, skipped


def ingest_source(source_name: str, limit: int = 200) -> RunResult:
    """Fetch one source into bronze, then transform just that run into silver."""
    module = sources.get(source_name)
    result = RunResult(source=source_name)
    with db.connect() as conn:
        run_id = db.start_run(conn, source_name)
        try:
            items = module.fetch(limit)
            result.fetched = db.insert_raw(conn, items, run_id)
            rows = db.fetch_raw(conn, run_id=run_id)
            postings, result.skipped = _to_postings(module, rows)
            result.inserted, result.updated = db.upsert_postings(conn, postings)
            db.finish_run(
                conn,
                run_id,
                status="ok",
                fetched=result.fetched,
                inserted=result.inserted,
                updated=result.updated,
                duplicates=db.count_duplicates(conn),
            )
        except Exception as exc:  # noqa: BLE001 - recorded, never raised
            result.failed = True
            result.error = str(exc)
            log.exception("ingest failed for %s", source_name)
            # The failure may have been the database itself, which leaves the
            # connection in an aborted transaction. Roll back first, or the
            # run-log update is refused too, the row stays 'running' forever,
            # and the error escapes to take the remaining sources down with it.
            conn.rollback()
            try:
                db.finish_run(conn, run_id, status="failed", error=str(exc))
            except Exception:  # noqa: BLE001
                log.exception("could not record the failed run for %s", source_name)
    return result


def transform(source_name: str | None = None) -> RunResult:
    """Re-parse everything in bronze. Use after changing a parser."""
    result = RunResult(source=source_name or "all")
    with db.connect() as conn:
        rows = db.fetch_raw(conn, source=source_name)
        by_source: dict[str, list[dict]] = {}
        for row in rows:
            by_source.setdefault(row["source"], []).append(row)
        for name, source_rows in by_source.items():
            module = sources.get(name)
            postings, skipped = _to_postings(module, source_rows)
            inserted, updated = db.upsert_postings(conn, postings)
            result.fetched += len(source_rows)
            result.inserted += inserted
            result.updated += updated
            result.skipped += skipped
    return result


def run(source_names: list[str] | None = None, limit: int = 200) -> list[RunResult]:
    """Ingest every requested source. One failing source does not stop the rest."""
    names = source_names or sources.DEFAULT_SOURCES
    results = []
    for source_name in names:
        log.info("ingesting %s", source_name)
        results.append(ingest_source(source_name, limit=limit))
    return results
