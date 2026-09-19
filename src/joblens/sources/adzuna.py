"""Adzuna.
Needs a free key from https://developer.adzuna.com. Set ADZUNA_APP_ID and
ADZUNA_APP_KEY in .env; without them this source is skipped rather than
failing the run, so the pipeline still works for anyone cloning the repo.
Adzuna is the only source of the three with structured salary fields, which
makes it the backbone of the Phase 2 regression model. Note that salaries
flagged salary_is_predicted are Adzuna's own estimate rather than something
the employer published, so we drop those: training a salary model on another
model's predictions is a good way to learn nothing.
"""

from __future__ import annotations

from datetime import datetime

from joblens.config import get_settings
from joblens.models import Posting, RawItem
from joblens.sources.base import client, get_json

name = "adzuna"
API_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
RESULTS_PER_PAGE = 50


def fetch(limit: int = 200) -> list[RawItem]:
    settings = get_settings()
    if not settings.adzuna_enabled:
        return []
    items: list[RawItem] = []
    pages = max(1, -(-limit // RESULTS_PER_PAGE))  # ceiling division
    with client() as http:
        for page in range(1, pages + 1):
            payload = get_json(
                http,
                API_URL.format(country=settings.adzuna_country, page=page),
                params={
                    "app_id": settings.adzuna_app_id,
                    "app_key": settings.adzuna_app_key,
                    "results_per_page": RESULTS_PER_PAGE,
                    "what": "machine learning engineer",
                    "content-type": "application/json",
                },
            )
            results = payload.get("results", [])
            if not results:
                break
            for entry in results:
                if not entry.get("id"):
                    continue
                items.append(
                    RawItem(source=name, source_id=str(entry["id"]), payload=entry)
                )
                if len(items) >= limit:
                    return items
    return items


def _salary_text(entry: dict) -> str | None:
    if entry.get("salary_is_predicted") in ("1", 1, True):
        return None
    low, high = entry.get("salary_min"), entry.get("salary_max")
    if low and high and low != high:
        return f"{low:,.0f} - {high:,.0f} per year"
    if low or high:
        return f"{(low or high):,.0f} per year"
    return None


def to_posting(payload: dict) -> Posting | None:
    title = payload.get("title")
    company = (payload.get("company") or {}).get("display_name")
    if not title or not company:
        return None
    posted_at = None
    if payload.get("created"):
        try:
            posted_at = datetime.fromisoformat(
                payload["created"].replace("Z", "+00:00")
            )
        except ValueError:
            posted_at = None
    return Posting.build(
        source=name,
        source_id=str(payload["id"]),
        title=title,
        company=company,
        url=payload.get("redirect_url", ""),
        location=(payload.get("location") or {}).get("display_name"),
        salary_raw=_salary_text(payload),
        description_html=payload.get("description"),
        posted_at=posted_at,
    )
