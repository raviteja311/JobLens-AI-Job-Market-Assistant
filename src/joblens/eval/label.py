"""Building judgements for the golden set.

Labelling is done against a pool, not against one retriever's output. If the
candidates come only from the system being evaluated, every posting it failed
to find is unjudged, unjudged counts as not relevant, and the system scores
100% against its own blind spots. Pooling every retriever is the standard fix
and it is the whole reason this module exists rather than a loop over
`search()`.

The pool is still not the corpus. A posting that no retriever surfaces is
never judged, so recall here is recall over the pool. At 465 postings the gap
is small; it would not be at 50,000, and the honest thing is to say so rather
than to quote the number as absolute.
"""

from __future__ import annotations

from dataclasses import dataclass

from joblens.eval.golden import GoldenQuery
from joblens.search import retrieval
from joblens.search.embeddings import Embedder

POOL_MODES = (
    {"mode": "keyword", "strategy": "whole"},
    {"mode": "vector", "strategy": "whole"},
    {"mode": "vector", "strategy": "section"},
)


@dataclass
class Candidate:
    key: str
    posting_id: int
    title: str
    company: str
    location: str | None
    is_remote: bool
    snippet: str
    found_by: list[str]
    grade: int = 0


def pool(
    conn,
    query: str,
    embedder: Embedder | None = None,
    depth: int = 10,
) -> list[Candidate]:
    """Top `depth` from every retriever, merged and de-duplicated."""
    seen: dict[int, Candidate] = {}
    for spec in POOL_MODES:
        hits = retrieval.search(
            conn,
            query,
            mode=spec["mode"],
            embedder=embedder,
            strategy=spec["strategy"],
            limit=depth,
        )
        for hit in hits:
            label = f"{spec['mode']}/{spec['strategy']}"
            if hit.posting_id in seen:
                seen[hit.posting_id].found_by.append(label)
            else:
                seen[hit.posting_id] = Candidate(
                    key="",
                    posting_id=hit.posting_id,
                    title=hit.title,
                    company=hit.company,
                    location=hit.location,
                    is_remote=hit.is_remote,
                    snippet=(hit.snippet or "")[:300],
                    found_by=[label],
                )
    if not seen:
        return []

    # The judgement key has to be the source's own id, not the primary key,
    # so the file survives a re-ingest into an empty database.
    rows = conn.execute(
        "select id, source, source_id from postings where id = any(%s)",
        (list(seen),),
    ).fetchall()
    for row in rows:
        seen[row["id"]].key = f"{row['source']}:{row['source_id']}"
    return list(seen.values())


def apply_judgements(
    query: GoldenQuery, graded: dict[str, int], verified: bool = True
) -> GoldenQuery:
    """Merge new grades into a query, keeping the zeros.

    Zeros are kept on purpose. "Judged and rejected" and "never looked at"
    are different states, and only the first one lets a later run tell that
    the pool has grown.
    """
    query.judgements.update({k: int(v) for k, v in graded.items()})
    query.verified = verified
    return query
