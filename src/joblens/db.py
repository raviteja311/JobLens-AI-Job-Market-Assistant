"""Database access.
Plain psycopg and plain SQL. No ORM: the queries here are simple, and being
able to read the exact SQL matters more than saving a few lines.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from joblens.config import get_settings
from joblens.models import Posting, RawItem

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


@contextmanager
def connect() -> Iterator[psycopg.Connection[dict]]:
    settings = get_settings()
    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        yield conn


def migrate() -> None:
    """Run every .sql file in migrations/ in filename order.
    Deliberately dumb. The files are written with `create table if not exists`
    so re-running is safe. If this project ever needs real migrations, swap in
    Alembic; right now that would be ceremony without benefit.
    """
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not files:
        raise RuntimeError(f"no migrations found in {MIGRATIONS_DIR}")
    with connect() as conn:
        for path in files:
            conn.execute(path.read_text())
        conn.commit()


def start_run(conn: psycopg.Connection, source: str) -> uuid.UUID:
    run_id = uuid.uuid4()
    conn.execute(
        "insert into ingestion_runs (run_id, source, status)"
        " values (%s, %s, 'running')",
        (run_id, source),
    )
    conn.commit()
    return run_id


def finish_run(
    conn: psycopg.Connection,
    run_id: uuid.UUID,
    *,
    status: str,
    fetched: int = 0,
    inserted: int = 0,
    updated: int = 0,
    duplicates: int = 0,
    error: str | None = None,
) -> None:
    conn.execute(
        """
        update ingestion_runs
           set finished_at = now(),
               status = %s,
               fetched = %s,
               inserted = %s,
               updated = %s,
               duplicates = %s,
               error = %s
         where run_id = %s
        """,
        (status, fetched, inserted, updated, duplicates, error, run_id),
    )
    conn.commit()


# Bronze keeps a posting's history, but only its changes. A board returns
# every open job on every run, and storing an identical copy of 24,000 jobs
# a day fills a free-tier database in days while recording nothing new.
#
# The check sends hashes, not payloads: shipping 24,000 payloads to Postgres
# just to compare them took almost two minutes, and their hashes take a
# fraction of a second. Only the payloads that changed are sent after it.
CHANGED_RAW_SQL = """
    select i.position
      from unnest(%(sources)s::text[], %(source_ids)s::text[], %(hashes)s::text[])
           with ordinality as i(source, source_id, hash, position)
     where i.hash is distinct from (
           select r.payload_hash
             from raw_postings r
            where r.source = i.source and r.source_id = i.source_id
            order by r.fetched_at desc, r.id desc
            limit 1)
"""


def payload_hash(payload: dict) -> str:
    """A stable hash of a payload. Sorted keys and fixed separators, so the
    same content always hashes the same whatever order the API sent it in."""
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


def insert_raw(
    conn: psycopg.Connection, items: Iterable[RawItem], run_id: uuid.UUID
) -> int:
    """Store the items that are new or changed since their last stored copy.
    Returns how many rows were written, which is 0 for a run that found
    nothing new."""
    items = list(items)
    if not items:
        return 0
    hashes = [payload_hash(i.payload) for i in items]
    changed = conn.execute(
        CHANGED_RAW_SQL,
        {
            "sources": [i.source for i in items],
            "source_ids": [i.source_id for i in items],
            "hashes": hashes,
        },
    ).fetchall()
    rows = []
    for row in changed:
        index = row["position"] - 1  # `with ordinality` counts from 1
        item = items[index]
        rows.append(
            (
                item.source,
                item.source_id,
                json.dumps(item.payload),
                hashes[index],
                run_id,
            )
        )
    if rows:
        with conn.cursor() as cur:
            cur.executemany(
                """
                insert into raw_postings
                    (source, source_id, payload, payload_hash, run_id)
                values (%s, %s, %s, %s, %s)
                """,
                rows,
            )
    conn.commit()
    return len(rows)


def fetch_raw(
    conn: psycopg.Connection,
    *,
    run_id: uuid.UUID | None = None,
    source: str | None = None,
) -> list[dict]:
    """Pull raw rows back out for transforming.
    With no run_id this returns the latest raw row per posting across all
    history, which is what you want when re-running the transform after a
    parser fix.
    """
    if run_id is not None:
        result = conn.execute(
            "select source, source_id, payload from raw_postings where run_id = %s",
            (run_id,),
        )
        return result.fetchall()

    sql = """
        select distinct on (source, source_id) source, source_id, payload
          from raw_postings
         {where}
         order by source, source_id, fetched_at desc
    """

    if source:
        result = conn.execute(sql.format(where="where source = %s"), (source,))
    else:
        result = conn.execute(sql.format(where=""))
    return result.fetchall()


UPSERT_SQL = """
insert into postings (
    source, source_id, title, company, location, is_remote,
    salary_raw, salary_min, salary_max, salary_period,
    salary_currency,
    salary_min_year, salary_max_year,
    description, url, posted_at, content_hash, last_seen_at
) values (
    %(source)s, %(source_id)s, %(title)s, %(company)s, %(location)s, %(is_remote)s,
    %(salary_raw)s, %(salary_min)s, %(salary_max)s, %(salary_period)s,
    %(salary_currency)s,
    %(salary_min_year)s, %(salary_max_year)s,
    %(description)s, %(url)s, %(posted_at)s, %(content_hash)s, now()
)
on conflict (source, source_id) do update
   set title           = excluded.title,
       company         = excluded.company,
       location        = excluded.location,
       is_remote       = excluded.is_remote,
       salary_raw      = excluded.salary_raw,
       salary_min      = excluded.salary_min,
       salary_max      = excluded.salary_max,
       salary_period   = excluded.salary_period,
       salary_currency = excluded.salary_currency,
       salary_min_year = excluded.salary_min_year,
       salary_max_year = excluded.salary_max_year,
       description     = excluded.description,
       url             = excluded.url,
       posted_at       = excluded.posted_at,
       content_hash    = excluded.content_hash,
       last_seen_at    = now()
returning (xmax = 0) as inserted
"""


def upsert_postings(
    conn: psycopg.Connection, postings: Iterable[Posting]
) -> tuple[int, int]:
    """Insert or refresh postings. Returns (inserted, updated).
    The `xmax = 0` trick tells us which branch Postgres took: xmax is zero on
    a fresh insert and non-zero when the row was updated by the conflict
    clause. It is the cheapest way to get real counts out of an upsert.
    """
    inserted = updated = 0
    with conn.cursor() as cur:
        for posting in postings:
            cur.execute(UPSERT_SQL, posting.model_dump())
            row = cur.fetchone()
            if row and row["inserted"]:
                inserted += 1
            else:
                updated += 1
    conn.commit()
    return inserted, updated


def count_duplicates(conn: psycopg.Connection) -> int:
    """Postings whose content_hash is shared with at least one other row."""
    result = conn.execute("""
        select coalesce(sum(c - 1), 0) as dupes
          from (select count(*) as c from postings
                 group by content_hash having count(*) > 1) t
        """)
    row = result.fetchone()
    return int(row["dupes"]) if row else 0
