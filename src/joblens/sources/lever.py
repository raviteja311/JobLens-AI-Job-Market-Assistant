"""Lever public postings API.

Official and keyless: https://github.com/lever/postings-api
One GET per company board in data/companies_in.yaml; the shared loop is in
sources/boards.py. The response is a plain JSON list of postings.

Quirks, ported from EasyJobs/lib/ats/lever.ts:
- A posting is split into an intro (description), titled bullet lists
  (lists[].text and lists[].content) and a closing note (additional). They
  are stitched back into one document, or the requirements are lost.
- createdAt is epoch milliseconds, not an ISO string.
- A multi-city job lists every city in categories.allLocations.
"""

from __future__ import annotations

from datetime import datetime, timezone

from joblens.models import Posting, RawItem
from joblens.sources.boards import SLUG_KEY, company_of, fetch_boards
from joblens.sources.filters import is_india, is_target_role

name = "lever"
BOARD_URL = "https://api.lever.co/v0/postings/{slug}"


def _location(job: dict) -> str | None:
    categories = job.get("categories") or {}
    every = [place for place in categories.get("allLocations") or [] if place]
    if every:
        return "; ".join(every)
    return (categories.get("location") or "").strip() or None


def _description_html(job: dict) -> str:
    parts = [job.get("description") or ""]
    for section in job.get("lists") or []:
        parts.append(
            f"<h3>{section.get('text') or ''}</h3>{section.get('content') or ''}"
        )
    parts.append(job.get("additional") or "")
    return "\n".join(part for part in parts if part)


def _posted_at(job: dict) -> datetime | None:
    created = job.get("createdAt")
    if not isinstance(created, (int, float)):
        return None
    return datetime.fromtimestamp(created / 1000, tz=timezone.utc)


def fetch(limit: int = 200) -> list[RawItem]:
    return fetch_boards(
        name,
        BOARD_URL,
        params={"mode": "json"},
        jobs_of=lambda response: response if isinstance(response, list) else [],
        title_of=lambda job: job.get("text"),
        location_of=_location,
        limit=limit,
    )


def to_posting(payload: dict) -> Posting | None:
    title = payload.get("text")
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
        url=payload.get("hostedUrl") or payload.get("applyUrl") or "",
        location=location,
        description_html=_description_html(payload),
        posted_at=_posted_at(payload),
        remote_hint=payload.get("workplaceType") == "remote",
    )
