"""The Indian company list and the script that exports it from EasyJobs."""

import importlib.util
from pathlib import Path

import httpx

from joblens.sources import companies

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "export_easyjobs_companies.py"
)


def load_script():
    spec = importlib.util.spec_from_file_location("export_companies", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_provider_has_companies():
    for provider in ("greenhouse", "lever", "ashby"):
        assert companies.for_provider(provider), provider


def test_boards_are_unique_and_complete():
    entries = [
        c for p in ("greenhouse", "lever", "ashby") for c in companies.for_provider(p)
    ]
    keys = [(c.provider, c.slug.lower()) for c in entries]
    assert len(keys) == len(set(keys))
    assert all(c.name and c.slug and c.city for c in entries)


def test_by_slug_finds_the_company():
    company = companies.by_slug("greenhouse", "razorpaysoftwareprivatelimited")
    assert company is not None and company.name == "Razorpay"
    assert companies.by_slug("lever", "razorpaysoftwareprivatelimited") is None


def test_export_keeps_only_boards_with_a_public_api():
    seed = "\n".join(
        [
            '  atsProvider: "greenhouse" | "lever";',
            '  { name: "A \\"Co\\"", city: "Pune", '
            'atsProvider: "greenhouse", atsSlug: "a" },',
            '  { name: "B", city: "Noida", atsProvider: "keka", atsSlug: "b" },',
            '  { name: "C", city: "Delhi", atsProvider: "workable", atsSlug: "c" },',
            '  { name: "D", city: "Kochi", atsProvider: "ashby", atsSlug: "d" },',
        ]
    )
    assert load_script().parse(seed) == [
        {"name": 'A "Co"', "provider": "greenhouse", "slug": "a", "city": "Pune"},
        {"name": "D", "provider": "ashby", "slug": "d", "city": "Kochi"},
    ]


AUDIT = Path(__file__).resolve().parents[1] / "scripts" / "check_company_boards.py"


def load_audit():
    spec = importlib.util.spec_from_file_location("check_company_boards", AUDIT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _board(provider, slug):
    company = companies.by_slug(provider, slug)
    assert company is not None, slug
    return company


def _http_error(code):
    request = httpx.Request("GET", "https://example.test/board")
    response = httpx.Response(code, request=request)
    return httpx.HTTPStatusError(str(code), request=request, response=response)


def test_audit_counts_jobs_india_jobs_and_kept_jobs(monkeypatch):
    audit = load_audit()
    jobs = [
        {"id": 1, "title": "Data Scientist", "location": {"name": "Bengaluru"}},
        {"id": 2, "title": "Accountant", "location": {"name": "Bengaluru"}},
        {"id": 3, "title": "Data Scientist", "location": {"name": "London"}},
        # No id: fetch_boards never stores it, so it is not counted.
        {"title": "Data Scientist", "location": {"name": "Pune"}},
    ]
    monkeypatch.setattr(audit.base, "get_json", lambda *a, **k: {"jobs": jobs})
    [row] = audit.audit([_board("greenhouse", "razorpaysoftwareprivatelimited")])
    assert row["status"] == "ok"
    assert (row["jobs_total"], row["jobs_in_india"], row["jobs_kept"]) == (3, 2, 1)
    assert row["name"] == "Razorpay" and row["checked_at"]


def test_audit_reads_each_provider_the_way_its_source_does(monkeypatch):
    audit = load_audit()
    responses = {
        "hevodata": [
            {"id": "a", "text": "Data Engineer", "categories": {"location": "Pune"}}
        ],
        "skyflow": {
            "jobs": [
                # The country alone puts it in India, as ashby._in_india decides.
                {
                    "id": "b",
                    "title": "ML Engineer",
                    "location": "Remote",
                    "address": {"postalAddress": {"addressCountry": "India"}},
                },
                # Hidden from the company's own board: in India, never kept.
                {
                    "id": "c",
                    "title": "Data Analyst",
                    "location": "Bengaluru",
                    "isListed": False,
                },
            ]
        },
    }

    def fake_get_json(http, url, params=None):
        return responses[url.rsplit("/", 1)[-1]]

    monkeypatch.setattr(audit.base, "get_json", fake_get_json)
    lever, ashby = audit.audit(
        [_board("lever", "hevodata"), _board("ashby", "skyflow")]
    )
    assert (lever["status"], lever["jobs_total"], lever["jobs_kept"]) == ("ok", 1, 1)
    assert ashby["status"] == "ok"
    assert (ashby["jobs_total"], ashby["jobs_in_india"], ashby["jobs_kept"]) == (
        2,
        2,
        1,
    )


def test_audit_records_a_404_with_zero_counts(monkeypatch):
    audit = load_audit()

    def not_found(*args, **kwargs):
        raise _http_error(404)

    monkeypatch.setattr(audit.base, "get_json", not_found)
    [row] = audit.audit([_board("lever", "hevodata")])
    assert row["status"] == "http_404"
    assert (row["jobs_total"], row["jobs_in_india"], row["jobs_kept"]) == (0, 0, 0)


def test_audit_records_a_connection_failure_as_error(monkeypatch):
    audit = load_audit()

    def unreachable(*args, **kwargs):
        # What base.get_json raises once its retries on a connection error
        # are spent.
        raise RuntimeError("giving up") from httpx.ConnectError("refused")

    monkeypatch.setattr(audit.base, "get_json", unreachable)
    [row] = audit.audit([_board("ashby", "skyflow")])
    assert row["status"] == "error"
    assert (row["jobs_total"], row["jobs_in_india"], row["jobs_kept"]) == (0, 0, 0)


def test_one_failing_board_does_not_stop_the_others(monkeypatch):
    audit = load_audit()
    job = {"id": 1, "title": "Data Scientist", "location": {"name": "Pune"}}

    def fake_get_json(http, url, params=None):
        if "/hevodata" in url:
            raise _http_error(500)
        if "/skyflow" in url:
            raise httpx.ConnectError("refused")
        return {"jobs": [job]}

    monkeypatch.setattr(audit.base, "get_json", fake_get_json)
    rows = audit.audit(
        [
            _board("lever", "hevodata"),
            _board("ashby", "skyflow"),
            _board("greenhouse", "razorpaysoftwareprivatelimited"),
        ]
    )
    assert [r["status"] for r in rows] == ["http_500", "error", "ok"]
    assert rows[2]["jobs_kept"] == 1
