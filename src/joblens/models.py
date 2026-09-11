"""The shapes that move through the pipeline.
RawItem is what a source hands us. Posting is what we store after cleaning.
Keeping them separate means a source can change its JSON without the rest of
the pipeline caring.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from joblens.cleaning import content_hash, is_remote, normalise_location, strip_html
from joblens.salary import parse_salary


class RawItem(BaseModel):
    """One posting exactly as the source gave it, plus the id we key on."""

    source: str
    source_id: str
    payload: dict


class Posting(BaseModel):
    source: str
    source_id: str
    title: str
    company: str
    location: str | None = None
    is_remote: bool = False
    salary_raw: str | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    salary_period: str | None = None
    salary_currency: str | None = None
    salary_min_year: float | None = None
    salary_max_year: float | None = None
    description: str = ""
    url: str
    posted_at: datetime | None = None
    content_hash: str = Field(default="")

    @field_validator("title", "company", mode="before")
    @classmethod
    def _require_text(cls, value: str | None) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("must not be empty")
        return text

    @classmethod
    def build(
        cls,
        *,
        source: str,
        source_id: str,
        title: str,
        company: str,
        url: str,
        location: str | None = None,
        salary_raw: str | None = None,
        description_html: str | None = None,
        posted_at: datetime | None = None,
        remote_hint: bool = False,
    ) -> Posting:
        """Apply every cleaning step in one place.
        Sources call this instead of constructing a Posting directly, so the
        cleaning rules cannot drift apart between one source and the next.
        """
        description = strip_html(description_html)
        remote = remote_hint or is_remote(location, title)
        clean_location = normalise_location(location)
        salary = parse_salary(salary_raw)
        annual_min, annual_max = salary.annualised()
        return cls(
            source=source,
            source_id=str(source_id),
            title=title.strip(),
            company=company.strip(),
            location=clean_location,
            is_remote=remote,
            salary_raw=salary_raw,
            salary_min=salary.min,
            salary_max=salary.max,
            salary_period=salary.period,
            salary_currency=salary.currency,
            salary_min_year=annual_min,
            salary_max_year=annual_max,
            description=description,
            url=url,
            posted_at=posted_at,
            content_hash=content_hash(title, company, clean_location),
        )
