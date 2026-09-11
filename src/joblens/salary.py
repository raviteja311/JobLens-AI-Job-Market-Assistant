from __future__ import annotations

import re
from dataclasses import dataclass

# Rough conversion factors. Not exact consistent, Which is what matters
# for comparing postings against each other.

HOURS_PER_YEAR = 2080
WEEKS_PER_YEAR = 52
MONTHS_PER_YEAR = 12
WORK_DAYS_PER_YEAR = 260


PERIOD_FACTORS = {
    "hour": HOURS_PER_YEAR,
    "day": WORK_DAYS_PER_YEAR,
    "week": WEEKS_PER_YEAR,
    "month": MONTHS_PER_YEAR,
    "year": 1,
}


CURRENCY_SYMBOLS = {
    "$": "USD",
    "£": "GBP",
    "€": "EUR",
    "₹": "INR",
    "¥": "JPY",
    "C$": "CAD",
    "A$": "AUD",
}

CURRENCY_CODES = {"USD", "GBP", "EUR", "INR", "CAD", "AUD", "JPY", "CHF", "SEK"}


PERIOD_PATTERNS = [
    ("hour", r"\b(?:per\s+hour|an?\s+hour|hourly|/\s*hro?u?r?|/\s*h\b|p\.?h\.?\b)"),
    ("day", r"\b(?:per\s+day|a\s+day|daily|/\s*day)"),
    ("week", r"\b(?:per\s+week|a\s+week|weekly|/\s*w(?:k|eek)?\b)"),
    ("month", r"\b(?:per\s+month|a\s+month|monthly|/\s*mo(?:nth)?\b|p\.?m\.?\b)"),
    (
        "year",
        r"\b(?:per\s+(?:year|annum)|an?\s+year|annually|annual|yearly"
        r"|p\.?a\.?\b|/\s*y(?:r|ear)?\b)",
    ),
]

NUMBER = r"\d{1,3}(?:[,\s]\d{2,3})+(?:\.\d+)?\s*[kK]?|\d+(?:\.\d+)?\s*[kK]?"
RANGE_SEPARATOR = r"\s*(?:--|-|–|—|to|and)\s*[\$£€₹¥]?\s*"
UPPER_BOUND_ONLY = re.compile(r"\bup\s+to\b", re.I)
LOWER_BOUND_ONLY = re.compile(r"\b(?:from|starting\s+at|upwards\s+of|\+)\b", re.I)
# Strings that mean "we are not telling you".
NON_NUMERIC_NOISE = re.compile(
    r"\b(?:competitive|negotiable|doe|depending\s+on\s+experience|market\s+rate|tbd)\b",
    re.I,
)


@dataclass(frozen=True)
class Salary:
    min: float | None = None
    max: float | None = None
    period: str | None = None
    currency: str | None = None

    @property
    def parsed(self) -> bool:
        return self.min is not None or self.max is not None

    def annualised(self) -> tuple[float | None, float | None]:
        """Both bounds converted to a yearly figure, for comparing postings
        Returns (None, None) if we nevwe worked out the period, because
        multiplying by a guessed factor is how you end up with a $60/hr
        contract sitting in the data as a $6o salary.
        """
        if not self.period:
            return (None, None)

        factor = PERIOD_FACTORS[self.period]
        lo = round(self.min * factor, 2) if self.min is not None else None
        hi = round(self.max * factor, 2) if self.max is not None else None
        return (lo, hi)


def _to_number(raw: str) -> float:
    token = raw.strip()
    multiplier = 1.0
    if token[-1] in "kK":
        multiplier = 1000.0
        token = token[:-1].strip()
    token = token.replace(",", "").replace(" ", "")
    return float(token) * multiplier


def _detect_currency(text: str) -> str | None:
    for symbol in ("C$", "A$"):
        if symbol in text:
            return CURRENCY_SYMBOLS[symbol]
    for symbol, code in CURRENCY_SYMBOLS.items():
        if len(symbol) == 1 and symbol in text:
            return code
    upper = text.upper()
    for code in CURRENCY_CODES:
        if re.search(rf"\b{code}\b", upper):
            return code
    return None


def _detect_period(text: str) -> str | None:
    lowered = text.lower()
    for period, pattern in PERIOD_PATTERNS:
        if re.search(pattern, lowered):
            return period
    return None


def _infer_period(value: float) -> str:
    """
    Fallback when the text does not sat.

    A bare 65 in a salary field is an hourly rate; a bare 130000 is annual.
    The cutoff is arbitrary but it is right far more often than it is wrong,
    and salary_raw stays in the database so we can audit it later.
    """
    return "hour" if value < 1000 else "year"


def parse_salary(text: str | None) -> Salary:
    """Pull a salary range out of free text. Returns an empty Salary if we cannot."""
    if not text:
        return Salary()
    cleaned = text.strip()
    if not re.search(r"\d", cleaned):
        return Salary()

    if NON_NUMERIC_NOISE.search(cleaned) and not re.search(
        r"[\$£€₹]|\d{4,}|\d+\s*[kK]\b", cleaned
    ):
        return Salary()

    body = re.split(r"\bplus\b|\+", cleaned, maxsplit=1)[0]

    currency = _detect_currency(cleaned)
    period = _detect_period(cleaned)

    range_match = re.search(rf"({NUMBER}){RANGE_SEPARATOR}({NUMBER})", body)

    if range_match and not UPPER_BOUND_ONLY.match(body.strip()):
        low = _to_number(range_match.group(1))
        high = _to_number(range_match.group(2))

        if high < 1000 <= low:
            high *= 1000
        if low > high:
            low, high = high, low
        return Salary(low, high, period or _infer_period(high), currency)

    single = re.search(rf"({NUMBER})", body)
    if not single:
        return Salary()
    value = _to_number(single.group(1))
    period = period or _infer_period(value)
    if UPPER_BOUND_ONLY.search(cleaned):
        return Salary(None, value, period, currency)
    if LOWER_BOUND_ONLY.search(cleaned):
        return Salary(value, None, period, currency)
    return Salary(value, value, period, currency)
