"""The Indian company list behind the Greenhouse, Lever and Ashby sources.

Those boards are per-company: there is no "search every Greenhouse job"
endpoint, so a source needs to know which boards to ask. The list lives in
data/companies_in.yaml, exported from the EasyJobs seed by
scripts/export_easyjobs_companies.py.

The company name a posting is stored under comes from this file, not from
the API, because the boards either omit it or return a legal entity name.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

COMPANIES_FILE = Path(__file__).resolve().parents[3] / "data" / "companies_in.yaml"


@dataclass(frozen=True)
class Company:
    name: str
    provider: str
    slug: str
    city: str


@lru_cache(maxsize=1)
def _load(path: Path) -> tuple[Company, ...]:
    entries = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return tuple(Company(**entry) for entry in entries)


def for_provider(provider: str, path: Path = COMPANIES_FILE) -> list[Company]:
    """Every company whose board runs on `provider`, in file order."""
    return [c for c in _load(path) if c.provider == provider]


def by_slug(provider: str, slug: str, path: Path = COMPANIES_FILE) -> Company | None:
    """The company behind one board, for to_posting() to recover the name
    from a source_id of the form "{slug}:{job_id}"."""
    for company in _load(path):
        if company.provider == provider and company.slug == slug:
            return company
    return None
