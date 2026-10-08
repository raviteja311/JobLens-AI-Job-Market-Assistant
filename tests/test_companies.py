"""The Indian company list and the script that exports it from EasyJobs."""

import importlib.util
from pathlib import Path

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
