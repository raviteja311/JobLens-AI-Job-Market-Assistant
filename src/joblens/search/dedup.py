"""Near-duplicate detection, the Phase 3 upgrade to the Phase 1 content hash.

The hash catches a job posted twice with identical text. It does not catch
the same job reposted a month later with a reworded title, or one company
posting to both boards with a different description. Those are the ones that
make a search results page look broken.

Embedding similarity catches them, at the cost of a threshold that has to be
chosen rather than derived. The number below was picked by looking at the
pairs it produces at several thresholds; `candidates()` exists so that
inspection can be repeated rather than taken on faith.

Verdicts go in their own table. The hash baseline stays exactly as it was, so
the two counts can be compared, which is the whole point of having kept it.
"""

from __future__ import annotations

from dataclasses import dataclass

from joblens import db

# Above this cosine similarity, two postings are the same job. Below 0.90 the
# pairs are genuinely different roles at the same company; above 0.97 it only
# finds what the hash already found.
DEFAULT_THRESHOLD = 0.93
METHOD = "embedding-whole"


@dataclass
class DuplicatePair:
    posting_id: int
    duplicate_of: int
    similarity: float
    title: str
    other_title: str
    company: str
    other_company: str


def candidates(
    threshold: float = DEFAULT_THRESHOLD,
    model: str = "all-MiniLM-L6-v2",
    limit: int = 500,
) -> list[DuplicatePair]:
    """Pairs above the threshold, newest posting first.

    Only the 'whole' strategy: a section chunk matching another posting's
    section chunk means both listings ask for Kubernetes, not that they are
    the same job.
    """
    with db.connect() as conn:
        rows = conn.execute(
            """
            select a.posting_id as posting_id,
                   b.posting_id as duplicate_of,
                   1 - (a.embedding <=> b.embedding) as similarity,
                   pa.title as title, pb.title as other_title,
                   pa.company as company, pb.company as other_company
              from posting_chunks a
              join posting_chunks b
                on b.strategy = a.strategy
               and b.model = a.model
               and b.posting_id < a.posting_id
              join postings pa on pa.id = a.posting_id
              join postings pb on pb.id = b.posting_id
             where a.strategy = 'whole' and a.model = %s
               and 1 - (a.embedding <=> b.embedding) >= %s
             order by similarity desc
             limit %s
            """,
            (model, threshold, limit),
        ).fetchall()
    return [DuplicatePair(**row) for row in rows]


def record(pairs: list[DuplicatePair], method: str = METHOD) -> int:
    """Persist verdicts. Idempotent, so it can run after every ingest."""
    if not pairs:
        return 0
    with db.connect() as conn, conn.cursor() as cur:
        cur.executemany(
            """
            insert into posting_duplicates
                (posting_id, duplicate_of, similarity, method)
            values (%s, %s, %s, %s)
            on conflict (posting_id, duplicate_of, method)
            do update set similarity = excluded.similarity
            """,
            [(p.posting_id, p.duplicate_of, p.similarity, method) for p in pairs],
        )
        conn.commit()
    return len(pairs)


def compare_to_hash(threshold: float = DEFAULT_THRESHOLD) -> dict:
    """The number Phase 1 promised: how much better is this than the hash.

    Not a superset, which was the assumption going in. The two methods read
    different fields. The hash reads title, company and location with
    seniority words stripped, so it pairs "Senior ML Engineer, London" with
    "ML Engineer, London" at the same company even when the two descriptions
    have nothing in common. The embedding reads the description, so it pairs
    two postings that describe the same work under different titles.

    At the default threshold each finds roughly seven pairs the other misses.
    Neither is the truth; `embedding_only` and `hash_only` are both worth
    reading before trusting either.
    """
    pairs = candidates(threshold=threshold)
    found = {(p.duplicate_of, p.posting_id) for p in pairs}
    with db.connect() as conn:
        hash_rows = conn.execute("""
            select least(a.id, b.id) as lo, greatest(a.id, b.id) as hi
              from postings a join postings b
                on a.content_hash = b.content_hash and a.id < b.id
            """).fetchall()
    hash_pairs = {(r["lo"], r["hi"]) for r in hash_rows}
    return {
        "threshold": threshold,
        "hash_pairs": len(hash_pairs),
        "embedding_pairs": len(found),
        "both": len(hash_pairs & found),
        "embedding_only": len(found - hash_pairs),
        "hash_only": len(hash_pairs - found),
    }
