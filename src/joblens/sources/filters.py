"""Which postings belong in the Indian AI and data corpus.

Two pure functions the board sources call inside to_posting(), returning
None when either fails. Filtering at parse time rather than fetch time is
deliberate: the raw payload still lands in bronze, so changing a rule here
and running `joblens transform` re-filters everything without a network call.

Every pattern uses word boundaries (\\b). Without them "india" matches
"Indiana" and "Indianapolis", "ai" matches "Email" and "ml" matches "HTML".
"""

from __future__ import annotations

import re

# Cities and states an Indian posting names, with the spellings boards use
# interchangeably (Bangalore and Bengaluru, Gurgaon and Gurugram). A posting
# that only says "Pune" or "Pune, Maharashtra" never says India.
_INDIA_PLACES = [
    "india",
    "bengaluru",
    "bangalore",
    "hyderabad",
    "pune",
    "mumbai",
    "bombay",
    "chennai",
    "madras",
    "gurugram",
    "gurgaon",
    "noida",
    "delhi",
    "kolkata",
    "calcutta",
    "ahmedabad",
    "kochi",
    "cochin",
    "coimbatore",
    "jaipur",
    "indore",
    "chandigarh",
    "thiruvananthapuram",
    "trivandrum",
    "lucknow",
    "bhubaneswar",
    "nagpur",
    "visakhapatnam",
    "vizag",
    "mysuru",
    "mysore",
    "mangaluru",
    "vadodara",
    "surat",
    "bhopal",
    "goa",
    "karnataka",
    "telangana",
    "maharashtra",
    "tamil nadu",
    "haryana",
    "kerala",
    "gujarat",
    "west bengal",
    "uttar pradesh",
]
_INDIA = re.compile(r"\b(?:" + "|".join(_INDIA_PLACES) + r")\b", re.IGNORECASE)

# Places that share a name with an Indian one. Hyderabad is also in Sindh.
_NOT_INDIA = re.compile(r"\bpakistan\b", re.IGNORECASE)

# Title words that put a role in the AI, ML, data and analytics family.
# "analyst" is not on the list by itself: at an Indian fintech most analyst
# titles are business roles ("Analyst, Banking Alliances"), so an analyst
# counts only through a qualifier that is on the list, such as "data
# analyst", "BI analyst" or "quantitative analyst".
_TARGET_TERMS = [
    r"data",
    r"analytics",
    r"scientist",
    r"machine learning",
    r"deep learning",
    r"computer vision",
    r"artificial intelligence",
    r"ai",
    r"gen ?ai",
    r"ml",
    r"ml ?ops",
    r"nlp",
    r"llms?",
    r"python",
    r"bi",
    r"business intelligence",
    r"quant(?:itative)?",
]
_TARGET = re.compile(r"\b(?:" + "|".join(_TARGET_TERMS) + r")\b", re.IGNORECASE)

# Titles that contain a target word but are not the work this corpus is
# about. Checked first, so "Account Executive, AI Products" is excluded.
_EXCLUDED_TERMS = [
    r"data entry",
    r"data cent(?:er|re)",
    r"sales",
    r"account executive",
    r"business development",
    r"recruiter",
    r"talent acquisition",
]
_EXCLUDED = re.compile(r"\b(?:" + "|".join(_EXCLUDED_TERMS) + r")\b", re.IGNORECASE)


def is_india(location: str | None) -> bool:
    """True when the location names India or an Indian city or state.

    A bare "Remote" is False on purpose: on these boards it usually means
    remote within the US. "Remote - India" is True through the word India.
    A multi-location string such as "London; Bengaluru" is True, because the
    job can be done from India.
    """
    if not location:
        return False
    if _NOT_INDIA.search(location):
        return False
    return bool(_INDIA.search(location))


def is_target_role(title: str | None) -> bool:
    """True when the title is an AI, ML, data, analytics or Python role."""
    if not title:
        return False
    if _EXCLUDED.search(title):
        return False
    return bool(_TARGET.search(title))
