"""Hacker News "Who is hiring" threads, via the Algolia API.
No key, no rate limit worth worrying about. The value here is that every
posting is a free-text comment written by a human with no schema at all, so
this is the source that stress-tests the parsers. It is also the source that
produces the interesting failures worth writing about.
The convention in these threads is a title line like:
    Acme Corp | Senior ML Engineer | London, UK | REMOTE | 90k-120k GBP
so we split on pipes and work out which field is which by inspection rather
than by position, because plenty of posters use a different order.
"""

from __future__ import annotations

import re
from datetime import datetime

from joblens.cleaning import REMOTE_HINTS, strip_html
from joblens.models import Posting, RawItem
from joblens.sources.base import client, get_json

name = "hackernews"
SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"
ITEM_URL = "https://hn.algolia.com/api/v1/items/{item_id}"
SALARY_HINT = re.compile(r"[\$£€₹]|\b\d{2,3}\s*[kK]\b|\b\d{5,6}\b|salary|comp\b", re.I)
LOCATION_HINT = re.compile(r",|remote|onsite|on-site|hybrid|\b[A-Z]{2}\b")


def _find_thread_ids(limit_threads: int = 2) -> list[int]:
    """Most recent 'Ask HN: Who is hiring?' threads, newest first."""
    with client() as http:
        payload = get_json(
            http,
            SEARCH_URL,
            params={
                "query": "Ask HN: Who is hiring?",
                "tags": "story,author_whoishiring",
                "hitsPerPage": limit_threads,
            },
        )
    return [int(hit["objectID"]) for hit in payload.get("hits", [])]


def fetch(limit: int = 300) -> list[RawItem]:
    items: list[RawItem] = []
    with client() as http:
        for thread_id in _find_thread_ids():
            thread = get_json(http, ITEM_URL.format(item_id=thread_id))
            for comment in thread.get("children", []):
                # Deleted comments come back with a null text field.
                if not comment.get("text") or not comment.get("id"):
                    continue
                items.append(
                    RawItem(
                        source=name,
                        source_id=str(comment["id"]),
                        payload=comment,
                    )
                )
                if len(items) >= limit:
                    return items
    return items


def _split_header(text: str) -> list[str]:
    first_line = strip_html(text).split("\n", 1)[0]
    return [part.strip() for part in first_line.split("|") if part.strip()]


def to_posting(payload: dict) -> Posting | None:
    """Best-effort parse of one comment. Returns None when it is not a job post.
    Roughly a third of comments in these threads are replies, questions or
    otherwise not job postings. Dropping them silently is correct; counting
    how many we drop is how we know the parser is not quietly eating real ones.
    """
    text = payload.get("text") or ""
    parts = _split_header(text)
    if len(parts) < 2:
        return None
    company, title = parts[0], parts[1]
    if len(company) > 80 or len(title) > 120:
        # A prose paragraph that happened to contain a pipe character.
        return None
    location = None
    salary_raw = None
    for part in parts[2:]:
        if salary_raw is None and SALARY_HINT.search(part):
            salary_raw = part
        elif location is None and LOCATION_HINT.search(part):
            location = part
    posted_at = None
    if payload.get("created_at"):
        try:
            posted_at = datetime.fromisoformat(
                payload["created_at"].replace("Z", "+00:00")
            )
        except ValueError:
            posted_at = None
    return Posting.build(
        source=name,
        source_id=str(payload["id"]),
        title=title,
        company=company,
        url=f"https://news.ycombinator.com/item?id={payload['id']}",
        location=location,
        salary_raw=salary_raw,
        description_html=text,
        posted_at=posted_at,
        remote_hint=bool(REMOTE_HINTS.search(" ".join(parts))),
    )
