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
from fastapi.responses import JSONResponse, Response

from joblens import db, observability
from joblens.api import schemas
from joblens.config import get_settings
from joblens.llm.client import BackendUnavailable
from joblens.ml import dataset, trends
from joblens.rag import chat as chat_rag
from joblens.rag import resume as resume_rag
from joblens.search import rerank, retrieval
from joblens.search.embeddings import get_embedder

log = logging.getLogger(__name__)

MAX_RESUME_BYTES = 2 * 1024 * 1024

# Module-level so the default is not a call in the signature.
RESUME_FILE = File(...)

_state: dict = {}


def warm_reranker() -> None:
    """Load the cross-encoder now, so the first reranked search does not."""
    rerank._load()
    log.info("cross-encoder %s ready", rerank.CROSS_ENCODER_MODEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    observability.ensure_logging(settings.log_level, settings.log_format)
    # Resolved at scrape time, not bound now: the collector is registered
    # once per process and must follow db.connect if it is ever replaced.
    observability.register_database_collector(lambda: db.connect())
    embedder = get_embedder()
    embedder.encode(["warm up"])  # pay the model load before serving traffic
    _state["embedder"] = embedder
    log.info("embedder %s ready", embedder.name)
    if settings.rerank_warmup:
        warm_reranker()
    yield
    _state.clear()
    _trends_cache.clear()


app = FastAPI(
    title="JobLens",
    version="0.4.0",
    description="Semantic job search, grounded chat and resume matching.",
    lifespan=lifespan,
)


@app.middleware("http")
async def observe(request: Request, call_next):
    """One histogram sample and one JSON log line per request.

    Also the body size guard. FastAPI checks the resume against
    MAX_RESUME_BYTES after reading it; this rejects on the declared length
    before a byte of a 500MB upload is buffered.
    """
    settings = get_settings()
    watch = observability.Stopwatch()
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > settings.max_body_bytes:
        response: Response = JSONResponse(
            {"detail": f"request body must be under {settings.max_body_bytes} bytes"},
            status_code=413,
        )
    else:
        try:
            response = await call_next(request)
        except Exception:
            _observe(request, 500, watch.seconds)
            raise
    _observe(request, response.status_code, watch.seconds)
    return response


def _observe(request: Request, status: int, seconds: float) -> None:
    route = observability.route_label(request.scope)
    method = request.method
    observability.REQUESTS.labels(method, route, str(status)).inc()
    observability.REQUEST_LATENCY.labels(method, route).observe(seconds)
    if route == "/metrics":
        return  # the scraper every 15 seconds is not traffic worth a line each
    log.info(
        "request",
        extra={
            "method": method,
            "route": route,
            "path": request.url.path,
            "status": status,
            "duration_ms": round(seconds * 1000, 1),
            "client": request.client.host if request.client else None,
        },
    )


# A fixed window per client, in memory. Correct for one process and wrong the
# moment there are two, which is fine at this size and is the kind of thing
# that should be written down rather than discovered.
_hits: dict[str, deque] = defaultdict(deque)
# Above this many distinct callers, forget the ones whose window has expired.
# Without a sweep the dict only ever grows: one entry per IP that ever called.
_SWEEP_ABOVE = 1000


def rate_limit(request: Request) -> None:
    settings = get_settings()
    caller = request.client.host if request.client else "unknown"
    now = time.monotonic()
    if len(_hits) > _SWEEP_ABOVE:
        for stale in [k for k, w in _hits.items() if not w or now - w[-1] > 60]:
            del _hits[stale]
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
    """Liveness plus the three facts the runbook asks for first: how much
    data, whether last night's ingest ran, and which LLM is configured."""
    with db.connect() as conn:
        postings = conn.execute("select count(*) as n from postings").fetchone()["n"]
        chunks = conn.execute("select count(*) as n from posting_chunks").fetchone()[
            "n"
        ]
        last_run = conn.execute("""
            select source, status, finished_at, inserted
              from ingestion_runs
             where status <> 'running'
             order by started_at desc
             limit 1
            """).fetchone()
    settings = get_settings()
    return {
        "status": "ok",
        "version": app.version,
        "postings": postings,
        "chunks": chunks,
        "embedder": _state["embedder"].name if "embedder" in _state else None,
        "llm_backend": settings.llm_backend,
        "llm_enabled": settings.llm_enabled,
        "last_ingest": (
            {
                "source": last_run["source"],
                "status": last_run["status"],
                "finished_at": (
                    last_run["finished_at"].isoformat()
                    if last_run["finished_at"]
                    else None
                ),
                "inserted": last_run["inserted"],
            }
            if last_run
            else None
        ),
    }


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    payload, content_type = observability.metrics_payload()
    return Response(payload, media_type=content_type)


@app.get("/search", response_model=schemas.SearchResponse)
def search(
    q: str = Query(min_length=2, max_length=300),
    mode: str = Query("hybrid", pattern="^(keyword|vector|hybrid)$"),
    strategy: str | None = Query(None, pattern="^(whole|section)$"),
    limit: int = Query(10, ge=1, le=50),
    rerank: bool = False,
) -> schemas.SearchResponse:
    began = time.perf_counter()
    # No strategy in the query means the configured one, not always "whole".
    strategy = strategy or get_settings().chunk_strategy
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


# /trends loads every posting into pandas and takes seconds. The corpus
# changes once a day, so the summary is kept per window for a few minutes.
_trends_cache: dict[int, tuple[float, dict]] = {}


@app.get("/trends")
def get_trends(days: int = Query(90, ge=1, le=365)) -> dict:
    ttl = get_settings().trends_cache_seconds
    cached = _trends_cache.get(days)
    if cached and time.monotonic() - cached[0] < ttl:
        return cached[1]
    with db.connect() as conn:
        frame = dataset.load_postings(conn=conn)
    summary = trends.summary(frame, days=days)
    _trends_cache[days] = (time.monotonic(), summary)
    return summary


@app.post(
    "/chat",
    response_model=schemas.ChatResponse,
    dependencies=[Depends(rate_limit)],
)
def chat(request: schemas.ChatRequest) -> schemas.ChatResponse:
    settings = get_settings()
    if not settings.llm_enabled:
        raise HTTPException(503, "no LLM backend configured")
    try:
        with db.connect() as conn:
            answer = chat_rag.ask(
                conn,
                request.question,
                embedder=_state.get("embedder"),
                limit=request.limit,
            )
    except BackendUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
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
def match(file: UploadFile = RESUME_FILE) -> schemas.MatchResponse:
    """Sync on purpose, like /chat. This handler makes up to nine blocking
    LLM calls; as `async def` it ran on the event loop and every other
    request, /health included, waited behind it. A plain `def` runs in the
    threadpool instead."""
    settings = get_settings()
    if not settings.llm_enabled:
        raise HTTPException(503, "no LLM backend configured")

    data = file.file.read()
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

    try:
        with db.connect() as conn:
            report = resume_rag.match_resume(
                conn, text, embedder=_state.get("embedder")
            )
    except BackendUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
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
