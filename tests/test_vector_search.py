"""vector_search against a real HNSW index.

The section search used to come back short: the planner filtered on strategy
after the HNSW scan, which stops after about hnsw.ef_search candidates, so a
request for 50 postings returned about 26 (docs/experiments.md, 2026-10-08).
These tests force the index plan, as the planner chooses it on the real
table, and check that a filtered search still fills its LIMIT.
"""

import numpy as np
import pytest

from joblens import db
from joblens.models import Posting
from joblens.search import retrieval

pytestmark = pytest.mark.db

DIM = 384
SECTION_POSTINGS = 120
CHUNKS_PER_POSTING = 3
WHOLE_DISTRACTORS = 400


def _database_available() -> bool:
    try:
        with db.connect() as conn:
            conn.execute("select 1")
        return True
    except Exception:
        return False


if not _database_available():
    pytest.skip("no Postgres reachable at DATABASE_URL", allow_module_level=True)


class FixedEmbedder:
    """Every query lands on the same point, so results are deterministic."""

    name = "test-model"
    dimension = DIM

    def __init__(self, vector):
        self.vector = vector

    def encode(self, texts):
        return np.array([self.vector for _ in texts], dtype=np.float32)


def _literal(vector) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in vector) + "]"


@pytest.fixture
def chunks():
    """Section chunks for 120 postings, plus 400 'whole' chunks that crowd the
    index around the query, which is what made the post-filter starve."""
    rng = np.random.default_rng(7)
    db.migrate()
    query = rng.normal(size=DIM)
    query /= np.linalg.norm(query)
    with db.connect() as conn:
        conn.execute("truncate postings, raw_postings, ingestion_runs cascade")
        postings = [
            Posting.build(
                source="test",
                source_id=str(i),
                title=f"Role {i}",
                company=f"Company {i}",
                url=f"https://example.com/{i}",
            )
            for i in range(SECTION_POSTINGS)
        ]
        db.upsert_postings(conn, postings)
        ids = [
            r["id"]
            for r in conn.execute(
                "select id from postings where source = 'test' order by id"
            ).fetchall()
        ]
        rows = []
        for posting_id in ids:
            for index in range(CHUNKS_PER_POSTING):
                vector = rng.normal(size=DIM)
                rows.append((posting_id, "section", index, "chunk", _literal(vector)))
        # The distractors sit right next to the query, so the index's first
        # ef_search candidates are almost all 'whole' and get filtered out.
        for index in range(WHOLE_DISTRACTORS):
            vector = query + rng.normal(scale=0.01, size=DIM)
            rows.append((ids[index % len(ids)], "whole", index, "x", _literal(vector)))
        with conn.cursor() as cur:
            cur.executemany(
                "insert into posting_chunks"
                " (posting_id, strategy, chunk_index, content, model, embedding)"
                " values (%s, %s, %s, %s, 'test-model', %s::vector)",
                rows,
            )
        conn.commit()
    yield FixedEmbedder(query)
    with db.connect() as conn:
        conn.execute("truncate postings, raw_postings, ingestion_runs cascade")
        conn.commit()


def _force_hnsw(conn) -> None:
    for setting in ("enable_seqscan", "enable_bitmapscan", "enable_sort"):
        conn.execute(f"set {setting} = off")


def _search(embedder, limit):
    with db.connect() as conn:
        conn.prepare_threshold = None
        # Force the HNSW plan, as the planner picks it on the real table. With
        # sorting allowed, this small table is cheaper to filter by btree and
        # sort exactly, which can never come back short.
        _force_hnsw(conn)
        return retrieval.vector_search(
            conn, "any query", embedder=embedder, strategy="section", limit=limit
        )


def test_the_index_plan_is_really_used(chunks):
    with db.connect() as conn:
        _force_hnsw(conn)
        plan = " ".join(
            row["QUERY PLAN"]
            for row in conn.execute(
                "explain (costs off) select id from posting_chunks"
                " where strategy = 'section' and model = 'test-model'"
                " order by embedding <=> %s::vector limit 200",
                (_literal(chunks.vector),),
            ).fetchall()
        )
    assert "posting_chunks_embedding_idx" in plan


def test_a_filtered_section_search_fills_its_limit(chunks):
    hits = _search(chunks, limit=50)
    assert len(hits) == 50
    assert len({h.posting_id for h in hits}) == 50


def test_results_are_sorted_by_similarity(chunks):
    scores = [h.score for h in _search(chunks, limit=30)]
    assert scores == sorted(scores, reverse=True)


def test_the_setting_does_not_leak_out_of_the_search(chunks):
    with db.connect() as conn:
        _force_hnsw(conn)
        retrieval.vector_search(
            conn, "q", embedder=chunks, strategy="section", limit=10
        )
        conn.commit()
        value = conn.execute("show hnsw.iterative_scan").fetchone()
    assert value["hnsw.iterative_scan"] == "off"


def test_health_reports_the_pgvector_version():
    # The deploy runbook reads this to confirm the hosted database can run
    # iterative index scans, which need pgvector 0.8 or newer.
    from joblens.api import main as api

    version = api.health()["pgvector"]
    major, minor = (int(part) for part in version.split(".")[:2])
    assert (major, minor) >= (0, 8)
