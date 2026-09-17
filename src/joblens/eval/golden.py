"""The golden dataset: queries with judged postings.

This is the slowest part of the project to build and the only reason any
other number in the repo means anything. Without it, "the new retriever is
better" is a feeling.

Two decisions worth stating.

**Judgements key on (source, source_id), never on the posting's primary key.**
`postings.id` is a bigserial. Re-ingest into an empty database and every id
shifts, which would silently re-point every judgement at a different job. The
source's own id is the only identifier that survives a rebuild.

**Relevance is graded, not binary.** 2 means this is what the query asked
for; 1 means a reasonable person would accept it; 0 means no. The difference
between a retriever that puts the 2s first and one that puts the 1s first is
invisible to a binary metric and obvious to anyone using the product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

GOLDEN_DIR = Path(__file__).resolve().parents[3] / "data" / "golden"
RETRIEVAL_FILE = GOLDEN_DIR / "retrieval.yaml"

GRADES = {0: "not relevant", 1: "acceptable", 2: "exactly what was asked"}


@dataclass
class GoldenQuery:
    id: str
    query: str
    # "source:source_id" -> grade
    judgements: dict[str, int] = field(default_factory=dict)
    note: str = ""
    verified: bool = False

    @property
    def relevant_keys(self) -> set[str]:
        return {key for key, grade in self.judgements.items() if grade > 0}

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "query": self.query,
            "note": self.note,
            "verified": self.verified,
            "judgements": dict(sorted(self.judgements.items())),
        }


def load(path: Path | None = None) -> list[GoldenQuery]:
    path = path or RETRIEVAL_FILE
    if not path.exists():
        return []
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [
        GoldenQuery(
            id=entry["id"],
            query=entry["query"],
            judgements={k: int(v) for k, v in (entry.get("judgements") or {}).items()},
            note=entry.get("note", "") or "",
            verified=bool(entry.get("verified", False)),
        )
        for entry in payload.get("queries", [])
    ]


def save(queries: list[GoldenQuery], path: Path | None = None) -> Path:
    path = path or RETRIEVAL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "grades": GRADES,
        "queries": [q.as_dict() for q in queries],
    }
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, width=88, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def verified_only(queries: list[GoldenQuery]) -> list[GoldenQuery]:
    """Queries a human has actually checked. The eval reports on these."""
    return [q for q in queries if q.verified and q.relevant_keys]


def resolve(conn, queries: list[GoldenQuery]) -> dict[str, dict[int, int]]:
    """Turn (source, source_id) judgements into posting ids for this database.

    A judged posting that is no longer in the corpus is dropped rather than
    counted as a miss. Job boards expire listings; scoring a retriever on its
    failure to return a deleted row measures nothing.
    """
    wanted = {key for q in queries for key in q.judgements}
    if not wanted:
        return {}
    pairs = [tuple(key.split(":", 1)) for key in wanted]
    rows = conn.execute(
        """
        select id, source, source_id from postings
         where (source, source_id) in (
               select unnest(%s::text[]), unnest(%s::text[])
         )
        """,
        ([p[0] for p in pairs], [p[1] for p in pairs]),
    ).fetchall()
    by_key = {f"{r['source']}:{r['source_id']}": r["id"] for r in rows}

    resolved: dict[str, dict[int, int]] = {}
    for query in queries:
        mapped = {
            by_key[key]: grade
            for key, grade in query.judgements.items()
            if key in by_key
        }
        resolved[query.id] = mapped
    return resolved


def coverage_report(conn, queries: list[GoldenQuery]) -> dict:
    """How much of the golden set still points at postings we hold."""
    resolved = resolve(conn, queries)
    judged = sum(len(q.judgements) for q in queries)
    present = sum(len(m) for m in resolved.values())
    verified = verified_only(queries)
    return {
        "queries": len(queries),
        "verified_queries": len(verified),
        "judgements": judged,
        "resolvable": present,
        "missing": judged - present,
        "mean_relevant_per_query": (
            round(
                sum(len(q.relevant_keys) for q in verified) / len(verified),
                2,
            )
            if verified
            else 0.0
        ),
    }
