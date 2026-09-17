"""Three retrievers and the merge that beats all of them.

Keyword search knows exact terms. Ask it for "pgvector" or "Series B" or a
job title someone actually wrote and it finds the row; ask it for "remote
roles for people who like building infrastructure" and it finds nothing,
because none of those words appear in the posting.

Vector search is the mirror image. It answers the vague question well and
loses the rare token: an embedding of "pgvector" is mostly "database", so the
one posting that names it does not come first.

Hybrid keeps both and merges by reciprocal rank fusion, which needs no score
calibration between the two, only their orderings. That matters because
ts_rank_cd and cosine similarity are not on the same scale and never will be.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from joblens.search.embeddings import Embedder, get_embedder

# The constant from the original RRF paper. It damps the contribution of the
# top rank so one retriever being confident cannot overrule the other outright.
RRF_K = 60


@dataclass
class SearchHit:
    posting_id: int
    title: str
    company: str
    url: str
    location: str | None
    is_remote: bool
    score: float
    # Which retrievers found it and where they ranked it. Kept because
    # "why is this result here" is the first question anyone asks, and
    # because the eval harness reports per-retriever contribution.
    ranks: dict[str, int] = field(default_factory=dict)
    snippet: str = ""


_SELECT = """
    select p.id, p.title, p.company, p.url, p.location, p.is_remote
"""


def _hit(row: dict, score: float, source: str, rank: int) -> SearchHit:
    return SearchHit(
        posting_id=row["id"],
        title=row["title"],
        company=row["company"],
        url=row["url"],
        location=row["location"],
        is_remote=row["is_remote"],
        score=score,
        ranks={source: rank},
        snippet=row.get("snippet", "") or "",
    )


# Terms are ORed, not ANDed, and this is the single most important line in
# the file. websearch_to_tsquery and plainto_tsquery both AND every term, so
# "remote machine learning engineer working on LLMs" becomes
# remot & machin & learn & engin & work & llm and matches nothing at all: the
# keyword arm of the hybrid contributed zero candidates and the first version
# of this module shipped that way.
#
# Extracting the lexemes from to_tsvector and joining them with | gives OR
# semantics with correct stemming and stopword removal, and no escaping to get
# wrong. Precision is then ts_rank_cd's job, which is what it is for: a
# posting matching five of the terms outranks one matching two.
_OR_QUERY = """
    to_tsquery('english', coalesce(nullif(
        array_to_string(tsvector_to_array(to_tsvector('english', %s)), ' | '), ''
    ), 'zzzzznomatchzzzzz'))
"""


def keyword_search(conn, query: str, limit: int = 20) -> list[SearchHit]:
    """Postgres full-text search over the stored tsvector."""
    rows = conn.execute(
        f"""
        with q as (select {_OR_QUERY} as query)
        {_SELECT}, ts_rank_cd(p.search_vector, q.query) as score,
               ts_headline('english', p.description, q.query,
                           'MaxWords=30, MinWords=10, MaxFragments=1') as snippet
          from postings p, q
         where p.search_vector @@ q.query
         order by score desc
         limit %s
        """,
        (query, limit),
    ).fetchall()
    return [
        _hit(row, float(row["score"]), "keyword", i + 1) for i, row in enumerate(rows)
    ]


def vector_search(
    conn,
    query: str,
    embedder: Embedder | None = None,
    strategy: str = "whole",
    limit: int = 20,
    chunk_multiplier: int = 4,
) -> list[SearchHit]:
    """Nearest chunks, collapsed to their postings.

    Fetches more chunks than needed and then collapses, because with the
    section strategy one posting can occupy several of the top slots. The
    posting's score is its best chunk: a role that matches the query in one
    section is a match, and averaging would punish long postings.
    """
    embedder = embedder or get_embedder()
    vector = embedder.encode([query])[0]
    literal = "[" + ",".join(f"{v:.6f}" for v in vector) + "]"

    rows = conn.execute(
        f"""
        with nearest as (
            select posting_id, content,
                   1 - (embedding <=> %s::vector) as similarity
              from posting_chunks
             where strategy = %s and model = %s
             order by embedding <=> %s::vector
             limit %s
        ),
        best as (
            select distinct on (posting_id) posting_id, content, similarity
              from nearest
             order by posting_id, similarity desc
        )
        {_SELECT}, b.similarity as score, b.content as snippet
          from best b join postings p on p.id = b.posting_id
         order by score desc
         limit %s
        """,
        (literal, strategy, embedder.name, literal, limit * chunk_multiplier, limit),
    ).fetchall()
    return [
        _hit(row, float(row["score"]), "vector", i + 1) for i, row in enumerate(rows)
    ]


def reciprocal_rank_fusion(
    runs: dict[str, list[SearchHit]], limit: int = 20, k: int = RRF_K
) -> list[SearchHit]:
    """Merge ranked lists by rank, not by score.

    Deliberately ignores how confident each retriever was. ts_rank_cd returns
    something around 0.1 and cosine similarity something around 0.6, and any
    attempt to put those on one scale is a fudge factor that has to be retuned
    every time either side changes.
    """
    merged: dict[int, SearchHit] = {}
    scores: dict[int, float] = {}
    for source, hits in runs.items():
        for rank, hit in enumerate(hits, start=1):
            scores[hit.posting_id] = scores.get(hit.posting_id, 0.0) + 1.0 / (k + rank)
            if hit.posting_id in merged:
                merged[hit.posting_id].ranks[source] = rank
                if not merged[hit.posting_id].snippet:
                    merged[hit.posting_id].snippet = hit.snippet
            else:
                existing = SearchHit(
                    posting_id=hit.posting_id,
                    title=hit.title,
                    company=hit.company,
                    url=hit.url,
                    location=hit.location,
                    is_remote=hit.is_remote,
                    score=0.0,
                    ranks={source: rank},
                    snippet=hit.snippet,
                )
                merged[hit.posting_id] = existing

    for posting_id, score in scores.items():
        merged[posting_id].score = score
    return sorted(merged.values(), key=lambda h: h.score, reverse=True)[:limit]


def hybrid_search(
    conn,
    query: str,
    embedder: Embedder | None = None,
    strategy: str = "whole",
    limit: int = 20,
    candidates: int = 50,
) -> list[SearchHit]:
    """Both retrievers, fused. The default for /search."""
    runs = {
        "keyword": keyword_search(conn, query, limit=candidates),
        "vector": vector_search(
            conn, query, embedder=embedder, strategy=strategy, limit=candidates
        ),
    }
    return reciprocal_rank_fusion(runs, limit=limit)


def search(
    conn,
    query: str,
    mode: str = "hybrid",
    embedder: Embedder | None = None,
    strategy: str = "whole",
    limit: int = 20,
    rerank: bool = False,
) -> list[SearchHit]:
    """One entry point, so the eval harness scores exactly what /search serves."""
    if mode == "keyword":
        hits = keyword_search(conn, query, limit=limit if not rerank else limit * 3)
    elif mode == "vector":
        hits = vector_search(
            conn,
            query,
            embedder=embedder,
            strategy=strategy,
            limit=limit if not rerank else limit * 3,
        )
    elif mode == "hybrid":
        hits = hybrid_search(
            conn,
            query,
            embedder=embedder,
            strategy=strategy,
            limit=limit if not rerank else limit * 3,
        )
    else:
        raise ValueError(f"unknown mode {mode!r}. known: keyword, vector, hybrid")

    if rerank:
        from joblens.search.rerank import rerank_hits

        hits = rerank_hits(query, hits, limit=limit)
    return hits[:limit]
