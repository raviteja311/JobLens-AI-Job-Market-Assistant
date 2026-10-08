"""Greenhouse public job boards.

Official, keyless and per-company: one GET per board, for every greenhouse
company in data/companies_in.yaml (see sources/companies.py). The response is
{"jobs": [...], "meta": {...}}. Fields a Posting needs:

    jobs[].id, title, location.name, absolute_url, updated_at, and content,
    which is HTML-escaped (html.unescape it before it reaches Posting.build).

The parsing quirks were already solved in TypeScript in EasyJobs/lib/ats/greenhouse.ts;
port them rather than rediscovering them. A saved response for unit tests is
tests/fixtures/greenhouse_jobs.json.

TODO(phase 1): implement fetch() and to_posting(). Until then this source
raises NotImplementedError, which ingest records as a failed run without
stopping the other sources.
"""

from __future__ import annotations

from joblens.models import Posting, RawItem
from joblens.sources import companies  # noqa: F401 - for to_posting()
from joblens.sources.base import client, get_json  # noqa: F401 - for fetch()

name = "greenhouse"
BOARD_URL = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"
# Pause between boards. These are free APIs; be a polite client.
DELAY_SECONDS = 0.5


def fetch(limit: int = 200) -> list[RawItem]:
    """GET every board in companies.for_provider(name), sleeping DELAY_SECONDS
    between boards, until `limit` items. Request with params={"content": "true"}.

    to_posting() only receives the payload, not the RawItem, so store the
    board slug inside the payload (for example payload["_slug"] = slug).
    source_id is f"{slug}:{job_id}", so ids never collide across companies.
    """
    raise NotImplementedError("greenhouse.fetch: phase 1 TODO")


def to_posting(payload: dict) -> Posting | None:
    """One job payload to a Posting, or None to skip it.

    Company name comes from companies.by_slug(name, payload["_slug"]), not
    from the API. Return None when filters.is_india(location) or
    filters.is_target_role(title) fails; the raw payload is still in bronze,
    so a later `joblens transform` can re-apply different filters offline.
    """
    raise NotImplementedError("greenhouse.to_posting: phase 1 TODO")
