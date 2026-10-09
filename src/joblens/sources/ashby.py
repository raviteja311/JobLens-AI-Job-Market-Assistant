"""Ashby public job board API.

Official and keyless: https://developers.ashbyhq.com/docs/job-posting-api
One GET per company board in data/companies_in.yaml; the shared loop is in
sources/boards.py. The response is {"jobs": [...], "apiVersion": "1"}.

Quirks, ported from EasyJobs/lib/ats/ashby.ts:
- isListed false means the company hid the job from its own board. It is
  still in the response, so it is skipped here.
- Extra cities are in secondaryLocations, each as {"location": ...}.
- location is often a bare city, but address.postalAddress.addressCountry
  usually names the country, so it is added to the location the India
  filter reads.
"""

from __future__ import annotations

from joblens.models import Posting, RawItem
from joblens.sources.boards import SLUG_KEY, company_of, fetch_boards, parse_iso
from joblens.sources.filters import is_india, is_target_role

name = "ashby"
BOARD_URL = "https://api.ashbyhq.com/posting-api/job-board/{slug}"


def _location(job: dict) -> str | None:
    places = [job.get("location")]
    places += [extra.get("location") for extra in job.get("secondaryLocations") or []]
    places = [place.strip() for place in places if place and place.strip()]
    return "; ".join(places) or None


def _country(job: dict) -> str | None:
    address = (job.get("address") or {}).get("postalAddress") or {}
    return address.get("addressCountry")


def _in_india(job: dict) -> bool:
    return is_india(_location(job)) or is_india(_country(job))


def fetch(limit: int = 200) -> list[RawItem]:
    return fetch_boards(
        name,
        BOARD_URL,
        jobs_of=lambda response: response.get("jobs") or [],
        title_of=lambda job: job.get("title"),
        # The filter count in fetch_boards reads a location string, so the
        # country rides along in it.
        location_of=lambda job: "; ".join(
            part for part in (_location(job), _country(job)) if part
        )
        or None,
        limit=limit,
    )


def to_posting(payload: dict) -> Posting | None:
    title = payload.get("title")
    company = company_of(name, payload)
    if not title or not company or payload.get("isListed") is False:
        return None
    if not _in_india(payload) or not is_target_role(title):
        return None
    return Posting.build(
        source=name,
        source_id=f"{payload[SLUG_KEY]}:{payload['id']}",
        title=title,
        company=company,
        url=payload.get("jobUrl") or payload.get("applyUrl") or "",
        location=_location(payload),
        description_html=payload.get("descriptionHtml")
        or payload.get("descriptionPlain"),
        posted_at=parse_iso(payload.get("publishedAt")),
        remote_hint=bool(payload.get("isRemote")),
    )
