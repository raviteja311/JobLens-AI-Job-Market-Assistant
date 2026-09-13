"""RemoteOK.
Public JSON feed, no key needed. https://remoteok.com/api
Two quirks worth knowing: the first element of the array is a legal notice
rather than a job, and salary arrives as two integers instead of text, so we
rebuild a salary string for the parser to chew on. That keeps one code path
for salary across every source.
"""

from __future__ import annotations

from datetime import datetime

from joblens.models import Posting, RawItem
from joblens.sources.base import client, get_json

name = "remoteok"
API_URL = "https://remoteok.com/api"


def fetch(limit: int = 200) -> list[RawItem]:
    with client() as http:
        payload = get_json(http, API_URL)
    items: list[RawItem] = []
    for entry in payload:
        # The legal notice has no id. Skip it rather than special-casing index 0,
        # in case they ever move it. Everything else goes to bronze untouched,
        # even entries missing a title: to_posting decides what is usable, so a
        # later parser fix can still recover the row.
        job_id = entry.get("id")
        if not job_id:
            continue
        items.append(RawItem(source=name, source_id=str(job_id), payload=entry))
        if len(items) >= limit:
            break
    return items


def _salary_text(entry: dict) -> str | None:
    low, high = entry.get("salary_min"), entry.get("salary_max")
    if low and high and low != high:
        return f"${low:,} - ${high:,} per year"
    if low or high:
        return f"${(low or high):,} per year"
    return None


def _posted_at(entry: dict) -> datetime | None:
    raw = entry.get("date")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def to_posting(payload: dict) -> Posting | None:
    if not payload.get("position") or not payload.get("company"):
        return None
    return Posting.build(
        source=name,
        source_id=str(payload["id"]),
        title=payload["position"],
        company=payload["company"],
        url=payload.get("url") or f"https://remoteok.com/remote-jobs/{payload['id']}",
        location=payload.get("location"),
        salary_raw=_salary_text(payload),
        description_html=payload.get("description"),
        posted_at=_posted_at(payload),
        remote_hint=True,  # the entire board is remote work
    )
