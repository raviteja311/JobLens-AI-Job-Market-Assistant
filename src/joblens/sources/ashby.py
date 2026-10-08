"""Ashby public job board API.

Official, keyless and per-company: one GET per board, for every ashby
company in data/companies_in.yaml (see sources/companies.py). The response is
{"jobs": [...], "apiVersion": "1"}. Fields a Posting needs:

    jobs[].id, title, location, jobUrl, descriptionHtml, publishedAt and
    isRemote. address.postalAddress.addressCountry is often present too.

The parsing quirks were already solved in TypeScript in EasyJobs/lib/ats/ashby.ts;
port them rather than rediscovering them. A saved response for unit tests is
tests/fixtures/ashby_job_board.json.

TODO(phase 1): implement fetch() and to_posting(). Until then this source
raises NotImplementedError, which ingest records as a failed run without
stopping the other sources.
"""

from __future__ import annotations

from joblens.models import Posting, RawItem
from joblens.sources import companies  # noqa: F401 - for to_posting()
from joblens.sources.base import client, get_json  # noqa: F401 - for fetch()

name = "ashby"
BOARD_URL = "https://api.ashbyhq.com/posting-api/job-board/{slug}"
# Pause between boards. These are free APIs; be a polite client.
DELAY_SECONDS = 0.5


def fetch(limit: int = 200) -> list[RawItem]:
    """GET every board in companies.for_provider(name), sleeping DELAY_SECONDS
    between boards, until `limit` items. No query parameters are needed.

    to_posting() only receives the payload, not the RawItem, so store the
    board slug inside the payload (for example payload["_slug"] = slug).
    source_id is f"{slug}:{job_id}", so ids never collide across companies.
    """
    raise NotImplementedError("ashby.fetch: phase 1 TODO")


def to_posting(payload: dict) -> Posting | None:
    """One job payload to a Posting, or None to skip it.

    Company name comes from companies.by_slug(name, payload["_slug"]), not
    from the API. Return None when filters.is_india(location) or
    filters.is_target_role(title) fails; the raw payload is still in bronze,
    so a later `joblens transform` can re-apply different filters offline.
    """
    raise NotImplementedError("ashby.to_posting: phase 1 TODO")
