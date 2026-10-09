"""Greenhouse public job boards.

Official and keyless: https://developers.greenhouse.io/job-board.html
One GET per company board in data/companies_in.yaml; the shared loop is in
sources/boards.py. The response is {"jobs": [...], "meta": {...}}.

Quirk: with content=true the description arrives HTML-escaped ("&lt;p&gt;"),
so it is unescaped once before strip_html can see the tags. Ported from
EasyJobs/lib/ats/greenhouse.ts.
"""

from __future__ import annotations

import html

from joblens.models import Posting, RawItem
from joblens.sources.boards import SLUG_KEY, company_of, fetch_boards, parse_iso
from joblens.sources.filters import is_india, is_target_role

name = "greenhouse"
BOARD_URL = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"


def _location(job: dict) -> str | None:
    return ((job.get("location") or {}).get("name") or "").strip() or None


def fetch(limit: int = 200) -> list[RawItem]:
    return fetch_boards(
        name,
        BOARD_URL,
        params={"content": "true"},
        jobs_of=lambda response: response.get("jobs") or [],
        title_of=lambda job: job.get("title"),
        location_of=_location,
        limit=limit,
    )


def to_posting(payload: dict) -> Posting | None:
    title = payload.get("title")
    location = _location(payload)
    company = company_of(name, payload)
    if not title or not company:
        return None
    if not is_india(location) or not is_target_role(title):
        return None
    return Posting.build(
        source=name,
        source_id=f"{payload[SLUG_KEY]}:{payload['id']}",
        title=title,
        company=company,
        url=payload.get("absolute_url", ""),
        location=location,
        description_html=html.unescape(payload.get("content") or ""),
        # first_published is when the job went up; updated_at moves on edits.
        posted_at=parse_iso(
            payload.get("first_published") or payload.get("updated_at")
        ),
    )
