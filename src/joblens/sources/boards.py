"""The fetch loop the Greenhouse, Lever and Ashby sources share.

All three are per-company boards: one GET per company in
data/companies_in.yaml, each returning that company's open jobs. Only the
URL, the shape of the response and the field names differ, so those are the
arguments and the loop lives here once.

Two decisions worth knowing:

- `limit` counts usable jobs, the ones that pass the India and target-role
  filters, not raw ones. A board returns every job the company has open,
  most of them outside the corpus, and a raw count would spend a 2,000 limit
  on the first few dozen companies.
- Every job from a visited board still goes to bronze, filtered or not. The
  filters run again in to_posting(), so a changed rule plus `joblens
  transform` re-filters everything without a network call.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime

import httpx

from joblens.models import RawItem
from joblens.sources import companies
from joblens.sources.base import client, get_json
from joblens.sources.filters import is_india, is_target_role

log = logging.getLogger(__name__)

# to_posting() only receives the payload, so fetch() stores the board slug in
# it. The underscore marks the one key that did not come from the API.
SLUG_KEY = "_slug"


def fetch_boards(
    provider: str,
    board_url: str,
    *,
    jobs_of: Callable[[object], list[dict]],
    title_of: Callable[[dict], str | None],
    location_of: Callable[[dict], str | None],
    limit: int,
    params: dict | None = None,
    delay: float = 0.5,
) -> list[RawItem]:
    """GET every board for `provider` until `limit` usable jobs are found.

    board_url has a {slug} placeholder. jobs_of pulls the job list out of a
    response, title_of and location_of read one job, so the filters can
    count usable jobs here with the same rules to_posting() applies.

    A board that fails (renamed slug, 404, the API down after retries) is
    logged and skipped: one dead board must not lose the other 290.
    """
    items: list[RawItem] = []
    usable = 0
    failed: list[str] = []
    boards = companies.for_provider(provider)
    with client() as http:
        for index, company in enumerate(boards):
            if index:
                time.sleep(delay)
            try:
                response = get_json(
                    http, board_url.format(slug=company.slug), params=params
                )
            except (httpx.HTTPError, RuntimeError) as exc:
                log.warning("%s board %r failed: %s", provider, company.slug, exc)
                failed.append(company.slug)
                continue
            for job in jobs_of(response):
                if job.get("id") is None:
                    continue
                payload = {**job, SLUG_KEY: company.slug}
                items.append(
                    RawItem(
                        source=provider,
                        source_id=f"{company.slug}:{job['id']}",
                        payload=payload,
                    )
                )
                if is_india(location_of(job)) and is_target_role(title_of(job)):
                    usable += 1
            if usable >= limit:
                break
    log.info(
        "%s: %s jobs from %s boards, %s usable, %s boards failed",
        provider,
        len(items),
        len(boards) - len(failed),
        usable,
        len(failed),
    )
    return items


def company_of(provider: str, payload: dict) -> str | None:
    """The company name from data/companies_in.yaml, not from the API."""
    company = companies.by_slug(provider, payload.get(SLUG_KEY, ""))
    return company.name if company else None


def parse_iso(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
