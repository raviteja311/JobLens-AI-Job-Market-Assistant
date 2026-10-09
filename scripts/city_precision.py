"""How often a city query's results are actually in that city.

    python scripts/city_precision.py

For every golden query that names an Indian city, the share of the top 10
whose location is in that city, per retriever. It needs no relevance labels,
so it can be run before and after a change to how location is indexed. It is
a proxy, not relevance: a Pune posting in the wrong role still counts, and a
remote role that a Pune seeker would accept does not.
"""

from __future__ import annotations

import re

from joblens import db
from joblens.eval import golden
from joblens.search import retrieval
from joblens.search.embeddings import get_embedder

# Query spelling -> spellings a posting's location may use.
CITIES = {
    "bangalore": ("bangalore", "bengaluru"),
    "bengaluru": ("bangalore", "bengaluru"),
    "hyderabad": ("hyderabad",),
    "pune": ("pune",),
    "chennai": ("chennai",),
    "mumbai": ("mumbai",),
    "noida": ("noida",),
    "gurgaon": ("gurgaon", "gurugram"),
    "gurugram": ("gurgaon", "gurugram"),
    "delhi": ("delhi",),
}
CONFIGS = (
    ("keyword", "keyword", "whole"),
    ("vector (whole)", "vector", "whole"),
    ("hybrid (whole)", "hybrid", "whole"),
)


def city_of(query: str) -> str | None:
    for city in CITIES:
        if re.search(rf"\b{city}\b", query, re.IGNORECASE):
            return city
    return None


def main() -> int:
    queries = [(q.id, q.query, city_of(q.query)) for q in golden.load()]
    queries = [q for q in queries if q[2]]
    embedder = get_embedder()
    embedder.encode(["warm up"])
    totals = {name: [] for name, _, _ in CONFIGS}
    with db.connect() as conn:
        for qid, text, city in queries:
            cells = []
            for name, mode, strategy in CONFIGS:
                hits = retrieval.search(
                    conn,
                    text,
                    mode=mode,
                    embedder=embedder,
                    strategy=strategy,
                    limit=10,
                )
                share = (
                    sum(
                        any(s in (h.location or "").lower() for s in CITIES[city])
                        for h in hits
                    )
                    / len(hits)
                    if hits
                    else 0.0
                )
                totals[name].append(share)
                cells.append(f"{share:.1f}")
            print(f"{qid} {text[:44]:44} " + "  ".join(cells))
    print(
        "mean "
        + " " * 44
        + "  ".join(f"{sum(v) / len(v):.2f}" for v in totals.values())
    )
    print("columns: " + ", ".join(name for name, _, _ in CONFIGS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
