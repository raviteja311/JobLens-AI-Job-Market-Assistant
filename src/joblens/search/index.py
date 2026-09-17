"""Building the vector index: postings -> chunks -> embeddings -> pgvector.

Separate from retrieval on purpose. Indexing is a batch job that runs after
ingestion; retrieval runs per request and must never trigger an embed of the
corpus because someone changed a config flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from joblens import db
from joblens.search import chunking
from joblens.search.embeddings import Embedder, get_embedder

log = logging.getLogger(__name__)


@dataclass
class IndexResult:
    strategy: str
    model: str
    postings: int
    chunks: int
    seconds: float
    usd: float

    def summary(self) -> str:
        per = self.chunks / self.seconds if self.seconds else 0
        return (
            f"{self.strategy}/{self.model}: {self.chunks} chunks from "
            f"{self.postings} postings in {self.seconds:.1f}s "
            f"({per:.0f} chunks/s, ${self.usd:.4f})"
        )


def _to_pgvector(values) -> str:
    """pgvector's text input format. Cheaper than adding a driver adapter."""
    return "[" + ",".join(f"{v:.6f}" for v in values) + "]"


def build_index(
    strategy: str = "whole",
    embedder: Embedder | None = None,
    rebuild: bool = False,
    batch_size: int = 256,
) -> IndexResult:
    """Embed every posting that does not already have a vector.

    Incremental by default. Re-embedding 465 postings costs a minute and
    nothing else, but at 50,000 it would be the difference between a nightly
    job that finishes and one that does not, so the query only picks up work
    that is actually missing.
    """
    embedder = embedder or get_embedder()
    with db.connect() as conn:
        if rebuild:
            conn.execute(
                "delete from posting_chunks where strategy = %s and model = %s",
                (strategy, embedder.name),
            )
            conn.commit()

        rows = conn.execute(
            """
            select p.id, p.title, p.company, p.description
              from postings p
             where not exists (
                   select 1 from posting_chunks c
                    where c.posting_id = p.id
                      and c.strategy = %s
                      and c.model = %s
             )
             order by p.id
            """,
            (strategy, embedder.name),
        ).fetchall()

        if not rows:
            log.info("index is already up to date for %s/%s", strategy, embedder.name)
            return IndexResult(strategy, embedder.name, 0, 0, 0.0, 0.0)

        chunks = []
        for row in rows:
            chunks.extend(
                chunking.chunk(
                    strategy,
                    row["id"],
                    row["title"],
                    row["company"],
                    row["description"] or "",
                )
            )

        log.info("embedding %s chunks from %s postings", len(chunks), len(rows))
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = embedder.encode([c.content for c in batch])
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    insert into posting_chunks
                        (posting_id, strategy, chunk_index, content, model, embedding)
                    values (%s, %s, %s, %s, %s, %s)
                    on conflict (posting_id, strategy, model, chunk_index)
                    do update set content = excluded.content,
                                  embedding = excluded.embedding
                    """,
                    [
                        (
                            c.posting_id,
                            c.strategy,
                            c.index,
                            c.content,
                            embedder.name,
                            _to_pgvector(vector),
                        )
                        for c, vector in zip(batch, vectors, strict=True)
                    ],
                )
            conn.commit()

    usage = getattr(embedder, "usage", None)
    return IndexResult(
        strategy=strategy,
        model=embedder.name,
        postings=len(rows),
        chunks=len(chunks),
        seconds=usage.seconds if usage else 0.0,
        usd=usage.usd if usage else 0.0,
    )


def index_stats() -> list[dict]:
    with db.connect() as conn:
        return conn.execute("""
            select strategy, model,
                   count(*) as chunks,
                   count(distinct posting_id) as postings,
                   round(avg(length(content))) as avg_chars
              from posting_chunks
             group by strategy, model
             order by strategy, model
            """).fetchall()
