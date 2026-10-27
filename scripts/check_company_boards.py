"""Check every company board in data/companies_in.yaml against the live APIs.

    python scripts/check_company_boards.py
    python scripts/check_company_boards.py --out other.csv --delay 1

The list is exported from the EasyJobs seed, and nothing in JobLens checked
that its boards answer or that they list jobs in India. This does, with the
same requests and the same rules as ingestion: one GET per board through
base.get_json, the job list and payload built as sources/boards.py builds
them, and the production to_posting() applied unchanged. It only reads; no
database is touched.

One row per board, in data/companies_check.csv:

- status: ok, http_<code> for a 4xx the API refused, or error (a connection
  failure, or a 429 or 5xx that outlasted get_json's retries)
- jobs_total: jobs with an id, the ones fetch_boards would store in bronze
- jobs_in_india: those the India filter accepts (for Ashby, as
  ashby._in_india decides, so the country field counts)
- jobs_kept: those to_posting() turns into a posting

The slug goes into the URL as it is in the file, exactly as fetch_boards
sends it; fetch_boards does not URL-encode it, and neither does this.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NamedTuple

import httpx

from joblens.sources import ashby, base, companies, greenhouse, lever
from joblens.sources.boards import SLUG_KEY
from joblens.sources.companies import Company
from joblens.sources.filters import is_india

OUT = Path(__file__).resolve().parents[1] / "data" / "companies_check.csv"
FIELDS = [
    "name",
    "provider",
    "slug",
    "city",
    "status",
    "jobs_total",
    "jobs_in_india",
    "jobs_kept",
    "checked_at",
]


class Provider(NamedTuple):
    """What each source's fetch() passes to fetch_boards, plus the India rule
    its to_posting() applies."""

    module: object
    params: dict | None
    jobs_of: Callable[[object], list[dict]]
    in_india: Callable[[dict], bool]


PROVIDERS = {
    "greenhouse": Provider(
        greenhouse,
        {"content": "true"},
        lambda response: response.get("jobs") or [],
        lambda job: is_india(greenhouse._location(job)),
    ),
    "lever": Provider(
        lever,
        {"mode": "json"},
        lambda response: response if isinstance(response, list) else [],
        lambda job: is_india(lever._location(job)),
    ),
    "ashby": Provider(
        ashby,
        None,
        lambda response: response.get("jobs") or [],
        ashby._in_india,
    ),
}


def check_board(http: httpx.Client, company: Company) -> dict:
    """One board's row. A failure is recorded, never raised."""
    provider = PROVIDERS[company.provider]
    row = {
        "name": company.name,
        "provider": company.provider,
        "slug": company.slug,
        "city": company.city,
        "status": "ok",
        "jobs_total": 0,
        "jobs_in_india": 0,
        "jobs_kept": 0,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    url = provider.module.BOARD_URL.format(slug=company.slug)
    try:
        response = base.get_json(http, url, params=provider.params)
    except httpx.HTTPStatusError as exc:
        row["status"] = f"http_{exc.response.status_code}"
        return row
    except Exception:  # noqa: BLE001 - one bad board must not stop the audit
        # Wider than fetch_boards, which catches HTTPError and RuntimeError:
        # a board answering with something other than JSON would stop it.
        row["status"] = "error"
        return row
    try:
        for job in provider.jobs_of(response):
            if job.get("id") is None:
                continue
            payload = {**job, SLUG_KEY: company.slug}
            row["jobs_total"] += 1
            row["jobs_in_india"] += provider.in_india(payload)
            row["jobs_kept"] += provider.module.to_posting(payload) is not None
    except Exception:  # noqa: BLE001 - an unexpected response shape
        row.update(status="error", jobs_total=0, jobs_in_india=0, jobs_kept=0)
    return row


def audit(entries: list[Company], *, delay: float = 0.0) -> list[dict]:
    rows = []
    with base.client() as http:
        for index, company in enumerate(entries):
            if index and delay:
                time.sleep(delay)
            rows.append(check_board(http, company))
    return rows


def summary(rows: list[dict]) -> str:
    statuses = Counter(row["status"] for row in rows)
    lines = [f"boards checked: {len(rows)}"]
    lines += [f"  {status}: {n}" for status, n in sorted(statuses.items())]
    lines.append(
        "boards with at least one India job: "
        f"{sum(1 for row in rows if row['jobs_in_india'])}"
    )
    lines.append(
        "boards with at least one kept job: "
        f"{sum(1 for row in rows if row['jobs_kept'])}"
    )
    lines.append(f"total kept jobs: {sum(row['jobs_kept'] for row in rows)}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--delay", type=float, default=0.5, help="seconds between boards"
    )
    args = parser.parse_args()

    entries = [c for p in PROVIDERS for c in companies.for_provider(p)]
    rows = audit(entries, delay=args.delay)
    rows.sort(key=lambda row: (row["provider"], row["name"].lower()))
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(summary(rows))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
