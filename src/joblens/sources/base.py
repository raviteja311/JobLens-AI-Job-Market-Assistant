"""Shared plumbing for every source.
A source is any module with two functions:
    fetch(limit) -> list[RawItem]     hit the API, return payloads untouched
    to_posting(payload) -> Posting    turn one payload into a clean Posting
Keeping fetch and parse separate is what lets us re-parse everything already
in the bronze table after fixing a bug, without touching the network.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

import httpx

from joblens.config import get_settings
from joblens.models import Posting, RawItem

log = logging.getLogger(__name__)

# Longest a single backoff may sleep. A Retry-After of 3600 from a board is
# real, and honouring it inside a cron job with a 30 minute timeout is not.
MAX_RETRY_DELAY = 60.0


def _retry_delay(header: str | None, fallback: float) -> float:
    """Seconds to wait, from a Retry-After header that may be delta-seconds
    or an HTTP-date, or absent, or nonsense. Never raises."""
    if not header:
        return min(fallback, MAX_RETRY_DELAY)
    try:
        seconds = float(header)
        if math.isfinite(seconds):
            return min(max(seconds, 0.0), MAX_RETRY_DELAY)
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(header)
    except (TypeError, ValueError):
        return min(fallback, MAX_RETRY_DELAY)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    seconds = (when - datetime.now(timezone.utc)).total_seconds()
    return min(max(seconds, 0.0), MAX_RETRY_DELAY)


class Source(Protocol):
    name: str

    def fetch(self, limit: int) -> list[RawItem]: ...
    def to_posting(self, payload: dict) -> Posting | None: ...
def client() -> httpx.Client:
    settings = get_settings()
    return httpx.Client(
        headers={"User-Agent": settings.user_agent, "Accept": "application/json"},
        timeout=settings.request_timeout,
        follow_redirects=True,
    )


def get_json(
    http: httpx.Client,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    attempts: int = 3,
) -> Any:
    """GET with backoff on rate limits and server errors.
    Anything in the 4xx range other than 429 is our fault, so we fail loudly
    instead of retrying a request that will never succeed.
    """
    delay = 2.0
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = http.get(url, params=params)
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = _retry_delay(response.headers.get("Retry-After"), delay)
                log.warning(
                    "%s returned %s, retrying in %.0fs (attempt %s/%s)",
                    url,
                    response.status_code,
                    retry_after,
                    attempt,
                    attempts,
                )
                time.sleep(retry_after)
                delay *= 2
                continue
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as exc:
            last_error = exc
            log.warning(
                "request to %s failed: %s (attempt %s/%s)", url, exc, attempt, attempts
            )
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"giving up on {url} after {attempts} attempts") from last_error


__all__ = ["Posting", "RawItem", "Source", "client", "get_json"]
