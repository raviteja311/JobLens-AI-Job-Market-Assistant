"""Lever public postings API.

Official, keyless and per-company: one GET per board, for every lever
company in data/companies_in.yaml (see sources/companies.py). The response is
a JSON list of postings. Fields a Posting needs:

    id, text (the title), categories.location, hostedUrl, createdAt (epoch
    milliseconds), and the description: descriptionPlain plus lists[], where
    each list is a heading (`text`) and HTML bullet items (`content`).

The parsing quirks were already solved in TypeScript in EasyJobs/lib/ats/lever.ts;
port them rather than rediscovering them. A saved response for unit tests is
tests/fixtures/lever_postings.json.

TODO(phase 1): implement fetch() and to_posting(). Until then this source
raises NotImplementedError, which ingest records as a failed run without
stopping the other sources.
"""

from __future__ import annotations

from joblens.models import Posting, RawItem
from joblens.sources import companies  # noqa: F401 - for to_posting()
from joblens.sources.base import client, get_json  # noqa: F401 - for fetch()

name = "lever"
BOARD_URL = "https://api.lever.co/v0/postings/{slug}"
# Pause between boards. These are free APIs; be a polite client.
DELAY_SECONDS = 0.5


def fetch(limit: int = 200) -> list[RawItem]:
    """GET every board in companies.for_provider(name), sleeping DELAY_SECONDS
    between boards, until `limit` items. Request with params={"mode": "json"}.

    to_posting() only receives the payload, not the RawItem, so store the
    board slug inside the payload (for example payload["_slug"] = slug).
    source_id is f"{slug}:{job_id}", so ids never collide across companies.
    """
    raise NotImplementedError("lever.fetch: phase 1 TODO")


def to_posting(payload: dict) -> Posting | None:
    """One job payload to a Posting, or None to skip it.

    Company name comes from companies.by_slug(name, payload["_slug"]), not
    from the API. Return None when filters.is_india(location) or
    filters.is_target_role(title) fails; the raw payload is still in bronze,
    so a later `joblens transform` can re-apply different filters offline.
    """
    raise NotImplementedError("lever.to_posting: phase 1 TODO")
