"""Which Indian city a query names, and whether a posting is there.

Neither retriever can be relied on to honour a city in the query. Keyword
search ORs the terms, so "data engineer jobs in Pune" is data | engin | job |
pune and a long posting that says "data" forty times outranks one that says
"Pune" once. A sentence embedding of the query is mostly about the role. So
hybrid search reads the city out of the query itself and gives postings in
that city an extra vote in the fusion (retrieval.hybrid_search).

Spellings are grouped because boards and people use both: a query for
Bengaluru must match a posting in Bangalore.
"""

from __future__ import annotations

import re

CITY_SPELLINGS: tuple[tuple[str, ...], ...] = (
    ("bengaluru", "bangalore"),
    ("hyderabad",),
    ("pune",),
    ("mumbai", "bombay", "navi mumbai"),
    ("chennai", "madras"),
    ("gurugram", "gurgaon"),
    ("noida",),
    ("delhi", "new delhi"),
    ("kolkata", "calcutta"),
    ("ahmedabad",),
    ("kochi", "cochin"),
    ("coimbatore",),
    ("jaipur",),
    ("indore",),
    ("chandigarh",),
    ("thiruvananthapuram", "trivandrum"),
    ("lucknow",),
    ("mysuru", "mysore"),
    ("visakhapatnam", "vizag"),
)


def _pattern(spellings: tuple[str, ...]) -> re.Pattern:
    return re.compile(
        r"\b(?:" + "|".join(re.escape(s) for s in spellings) + r")\b", re.IGNORECASE
    )


_CITIES = [(spellings, _pattern(spellings)) for spellings in CITY_SPELLINGS]


def query_city(query: str) -> re.Pattern | None:
    """A pattern matching every spelling of the city the query names, or
    None. The first city wins when a query names several."""
    for _, pattern in _CITIES:
        if pattern.search(query or ""):
            return pattern
    return None


def in_city(location: str | None, city: re.Pattern) -> bool:
    return bool(location and city.search(location))
