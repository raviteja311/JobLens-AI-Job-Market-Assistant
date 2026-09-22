"""Phase 7: what the service says about itself.

Two outputs, both boring on purpose.

Logs are one JSON object per line when LOG_FORMAT=json, so a hosted log
search can filter on `route` or `status` instead of grepping prose. The
text format stays for a terminal, where JSON is unreadable.

Metrics are Prometheus text on /metrics. The request histogram and the LLM
counters are updated in-process as things happen. The gauges that describe
the database (postings, today's LLM spend, when each source last ingested
successfully) are read from Postgres at scrape time instead: a counter that
resets on every deploy cannot answer "did last night's ingest run", and
that is the question the on-call page exists to ask.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterator
from datetime import datetime, timezone

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Histogram,
    generate_latest,
)
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- logging

# Everything a LogRecord carries by default. Anything else on the record was
# passed through `extra=` and belongs in the JSON line as its own key.
_RESERVED = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "message",
    "module",
    "msecs",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "taskName",
    "thread",
    "threadName",
}

TEXT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


class JsonFormatter(logging.Formatter):
    """One object per line. Keys are stable so dashboards can rely on them."""

    def format(self, record: logging.LogRecord) -> str:
        line = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                line[key] = value
        if record.exc_info:
            line["exc"] = self.formatException(record.exc_info)
        return json.dumps(line, default=str, ensure_ascii=False)


def configure_logging(level: int | str = logging.INFO, fmt: str = "text") -> None:
    """Install one handler on the root logger. Idempotent."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(TEXT_FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(level)
    # uvicorn's own access line duplicates the one the middleware writes,
    # and its error logger propagates to root anyway.
    logging.getLogger("uvicorn.access").disabled = True


def ensure_logging(level: int | str = logging.INFO, fmt: str = "text") -> None:
    """Configure only if nobody has. For the uvicorn --reload child process,
    which imports the app without going through the CLI."""
    if not logging.getLogger().handlers:
        configure_logging(level, fmt)


# ---------------------------------------------------------------- metrics

# Buckets run to two minutes because /chat on a local model takes about
# fifty seconds, and a histogram whose top bucket is 10s would file every
# chat request under "more than that" and show nothing.
LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
)

REQUESTS = Counter(
    "joblens_http_requests_total",
    "HTTP requests served",
    ["method", "route", "status"],
)
REQUEST_LATENCY = Histogram(
    "joblens_http_request_duration_seconds",
    "Wall time per request, by route",
    ["method", "route"],
    buckets=LATENCY_BUCKETS,
)
LLM_CALLS = Counter(
    "joblens_llm_calls_total",
    "LLM completions attempted, by feature and outcome",
    ["feature", "backend", "model", "ok"],
)
LLM_COST = Counter(
    "joblens_llm_cost_usd_total",
    "USD spent on LLM calls since this process started",
    ["feature", "backend", "model"],
)
LLM_LATENCY = Histogram(
    "joblens_llm_call_duration_seconds",
    "Wall time per LLM call",
    ["feature", "backend"],
    buckets=LATENCY_BUCKETS,
)


def record_llm_call(
    *,
    feature: str,
    backend: str,
    model: str,
    ok: bool,
    cost_usd: float,
    latency_ms: int | None,
) -> None:
    LLM_CALLS.labels(feature, backend, model, str(ok).lower()).inc()
    if cost_usd:
        LLM_COST.labels(feature, backend, model).inc(float(cost_usd))
    if latency_ms is not None:
        LLM_LATENCY.labels(feature, backend).observe(latency_ms / 1000)


def route_label(scope: dict) -> str:
    """The route template, not the path. /search?q=... is one series, not one
    series per query, and a 404 for /wp-admin does not get to create one."""
    route = scope.get("route")
    path = getattr(route, "path", None)
    return path if path else "unmatched"


class DatabaseCollector(Collector):
    """Gauges read from Postgres at scrape time.

    A scrape must never fail because the database did: that is exactly the
    moment the dashboard is needed. When the query errors, the collector
    reports joblens_db_up 0 and nothing else.
    """

    def __init__(self, connect):
        self._connect = connect

    def collect(self) -> Iterator[GaugeMetricFamily]:
        up = GaugeMetricFamily("joblens_db_up", "1 if the last scrape query worked")
        try:
            with self._connect() as conn:
                postings = conn.execute(
                    "select count(*) as n from postings"
                ).fetchone()["n"]
                cost_today = conn.execute("""
                    select coalesce(sum(cost_usd), 0) as usd,
                           count(*) as calls
                      from llm_calls
                     where created_at >= date_trunc('day', now())
                    """).fetchone()
                runs = conn.execute("""
                    select distinct on (source)
                           source, status, started_at, finished_at, inserted
                      from ingestion_runs
                     where status <> 'running'
                     order by source, started_at desc
                    """).fetchall()
                last_ok = conn.execute("""
                    select source, max(finished_at) as finished_at
                      from ingestion_runs
                     where status = 'ok'
                     group by source
                    """).fetchall()
        except Exception as exc:  # noqa: BLE001 - the scrape reports, it does not raise
            log.warning("metrics scrape could not reach the database: %s", exc)
            up.add_metric([], 0)
            yield up
            return

        up.add_metric([], 1)
        yield up

        g = GaugeMetricFamily("joblens_postings", "Rows in the postings table")
        g.add_metric([], postings)
        yield g

        g = GaugeMetricFamily(
            "joblens_llm_cost_usd_today", "USD spent on LLM calls since midnight UTC"
        )
        g.add_metric([], float(cost_today["usd"]))
        yield g
        g = GaugeMetricFamily(
            "joblens_llm_calls_today", "LLM calls made since midnight UTC"
        )
        g.add_metric([], cost_today["calls"])
        yield g

        ok = GaugeMetricFamily(
            "joblens_ingest_last_run_ok",
            "1 if the most recent finished ingestion run for the source succeeded",
            labels=["source"],
        )
        inserted = GaugeMetricFamily(
            "joblens_ingest_last_run_inserted",
            "New postings from the most recent finished run",
            labels=["source"],
        )
        for run in runs:
            ok.add_metric([run["source"]], 1 if run["status"] == "ok" else 0)
            inserted.add_metric([run["source"]], run["inserted"])
        yield ok
        yield inserted

        success = GaugeMetricFamily(
            "joblens_ingest_last_success_timestamp_seconds",
            "When the source last ingested successfully. Alert when stale.",
            labels=["source"],
        )
        for row in last_ok:
            success.add_metric([row["source"]], row["finished_at"].timestamp())
        yield success


_registered: set[str] = set()


def register_database_collector(connect) -> None:
    """Idempotent: the app module can be imported more than once in tests."""
    if "db" in _registered:
        return
    REGISTRY.register(DatabaseCollector(connect))
    _registered.add("db")


def metrics_payload() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


class Stopwatch:
    def __init__(self) -> None:
        self.began = time.perf_counter()

    @property
    def seconds(self) -> float:
        return time.perf_counter() - self.began

    @property
    def ms(self) -> float:
        return round(self.seconds * 1000, 1)
