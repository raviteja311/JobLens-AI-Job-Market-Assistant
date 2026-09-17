"""The FastAPI backend.

Four endpoints. /search and /trends are free to serve. /chat and /match each
cost an LLM call, so they sit behind a rate limit: this is a personal project
whose whole point is to have a public URL, and a public URL with an uncapped
model call behind it is someone else's free inference endpoint.

The embedder is loaded once at startup rather than per request. Loading it
lazily inside the handler makes the first user after a deploy wait two
seconds and hides the cost from every latency graph.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile

from joblens import db
from joblens.api import schemas
from joblens.config import get_settings
from joblens.ml import dataset, trends
from joblens.rag import chat as chat_rag
from joblens.rag import resume as resume_rag
from joblens.search import retrieval
from joblens.search.embeddings import get_embedder

log = logging.getLogger(__name__)

MAX_RESUME_BYTES = 2 * 1024 * 1024

# Module-level so the default is not a call in the signature.
RESUME_FILE = File(...)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder = get_embedder()
    embedder.encode(["warm up"])  # pay the model load before serving traffic
    _state["embedder"] = embedder
    log.info("embedder %s ready", embedder.name)
    yield
    _state.clear()


app = FastAPI(
    title="JobLens",
    version="0.3.0",
    description="Semantic job search, grounded chat and resume matching.",
    lifespan=lifespan,
)


# A fixed window per client, in memory. Correct for one process and wrong the
# moment there are two, which is fine at this size and is the kind of thing
# that should be written down rather than discovered.
_hits: dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request) -> None:
    settings = get_settings()
    caller = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _hits[caller]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= settings.rate_limit_per_minute:
        raise HTTPException(
            status_code=429,
            detail=f"limit is {settings.rate_limit_per_minute} requests a minute",
        )
    window.append(now)


@app.get("/health")
def health() -> dict:
    with db.connect() as conn:
        postings = conn.execute("select count(*) as n from postings").fetchone()["n"]
        chunks = conn.execute("select count(*) as n from posting_chunks").fetchone()[
            "n"
        ]
    settings = get_settings()
    return {
        "status": "ok",
        "postings": postings,
        "chunks": chunks,
        "embedder": _state["embedder"].name if "embedder" in _state else None,
        "llm_backend": settings.llm_backend,
        "llm_enabled": settings.llm_enabled,
    }


@app.get("/search", response_model=schemas.SearchResponse)
def search(
    q: str = Query(min_length=2, max_length=300),
    mode: str = Query("hybrid", pattern="^(keyword|vector|hybrid)$"),
    strategy: str = Query("whole", pattern="^(whole|section)$"),
    limit: int = Query(10, ge=1, le=50),
    rerank: bool = False,
) -> schemas.SearchResponse:
    began = time.perf_counter()
    with db.connect() as conn:
        hits = retrieval.search(
            conn,
            q,
            mode=mode,
            embedder=_state.get("embedder"),
            strategy=strategy,
            limit=limit,
            rerank=rerank,
        )
    return schemas.SearchResponse(
        query=q,
        mode=mode,
        reranked=rerank,
        took_ms=round((time.perf_counter() - began) * 1000, 1),
        results=[
            schemas.SearchResult(
                posting_id=h.posting_id,
                title=h.title,
                company=h.company,
                url=h.url,
                location=h.location,
                is_remote=h.is_remote,
                score=round(h.score, 5),
                ranks=h.ranks,
                snippet=h.snippet,
            )
            for h in hits
        ],
    )


@app.get("/trends")
def get_trends(days: int = Query(90, ge=1, le=365)) -> dict:
    with db.connect() as conn:
        frame = dataset.load_postings(conn=conn)
    return trends.summary(frame, days=days)


@app.post(
    "/chat",
    response_model=schemas.ChatResponse,
    dependencies=[Depends(rate_limit)],
)
def chat(request: schemas.ChatRequest) -> schemas.ChatResponse:
    settings = get_settings()
    if not settings.llm_enabled:
        raise HTTPException(503, "no LLM backend configured")
    with db.connect() as conn:
        answer = chat_rag.ask(
            conn,
            request.question,
            embedder=_state.get("embedder"),
            limit=request.limit,
        )
    return schemas.ChatResponse(
        question=answer.question,
        answer=answer.answer,
        citations=[
            schemas.Citation(
                n=s.n,
                posting_id=s.posting_id,
                title=s.title,
                company=s.company,
                url=s.url,
            )
            for s in answer.sources
        ],
        grounded=answer.grounded,
        took_ms=answer.took_ms,
        cost_usd=round(answer.cost_usd, 6),
    )


@app.post(
    "/match",
    response_model=schemas.MatchResponse,
    dependencies=[Depends(rate_limit)],
)
async def match(file: UploadFile = RESUME_FILE) -> schemas.MatchResponse:
    settings = get_settings()
    if not settings.llm_enabled:
        raise HTTPException(503, "no LLM backend configured")

    data = await file.read()
    if len(data) > MAX_RESUME_BYTES:
        raise HTTPException(413, "resume must be under 2MB")
    try:
        text = (
            resume_rag.pdf_to_text(data)
            if (file.filename or "").lower().endswith(".pdf")
            else data.decode("utf-8", errors="replace")[: resume_rag.MAX_RESUME_CHARS]
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    with db.connect() as conn:
        report = resume_rag.match_resume(conn, text, embedder=_state.get("embedder"))
    return schemas.MatchResponse(
        resume_skills=report.profile.skills,
        matches=[
            schemas.MatchResult(
                posting_id=m.posting_id,
                title=m.title,
                company=m.company,
                url=m.url,
                fit_score=m.verdict.fit_score,
                reasoning=m.verdict.reasoning,
                matched_skills=m.verdict.matched_skills,
                missing_skills=m.verdict.missing_skills,
            )
            for m in report.matches
        ],
        took_ms=report.took_ms,
        cost_usd=round(report.cost_usd, 6),
    )
