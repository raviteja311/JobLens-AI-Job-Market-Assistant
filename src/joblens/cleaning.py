"""Cleaning helpers for turning source payloads into comparable fields."""

from __future__ import annotations

import hashlib
import re
from html import unescape
from html.parser import HTMLParser

# Tags whose contents we do not want in the description text at all.
_SKIP_TAGS = {"script", "style"}
# Tags that should leave a line break behind so paragraphs do not run together.
_BREAK_TAGS = {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}


class _TextExtractor(HTMLParser):

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BREAK_TAGS:
            self.chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _BREAK_TAGS:
            self.chunks.append("\n")

    def handle_data(self, data):
        if not self._skip_depth:
            self.chunks.append(data)


def strip_html(raw: str | None) -> str:
    """Turn a chunk of posting HTML into readable plain text.
    stdlib rather than BeautifulSoup: this is a small job and one less
    dependency in the deployed image. If descriptions get gnarlier than
    the boards currently serve, swap it out here and nothing else changes.
    """
    if not raw:
        return ""
    parser = _TextExtractor()
    parser.feed(raw)
    parser.close()
    text = unescape("".join(parser.chunks))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


URL = re.compile(r"https?://\S+|\bwww\.\S+", re.I)

# Board furniture that appears in every posting from a source and therefore
# carries no information about the job.
#
# The RemoteOK one is worth reading. Every posting it serves ends with a line
# asking the applicant to quote a magic word and a token, where the token is
# base64 of the IP address that fetched the page. It is an anti-bot canary, so
# it is identical across the whole source and unique against every other
# source, which made it the single strongest term in the corpus: k-means
# happily produced a cluster that meant "came from RemoteOK" and we nearly
# shipped it as a job family. It is also a sentence of instructions living
# inside scraped text, which is worth remembering in Phase 4 when that text
# starts going into prompts.
_BOILERPLATE = (
    re.compile(r"please mention the word\b.*?(?:\.|$)", re.I | re.S),
    re.compile(r"this is a beta feature to avoid spam applicants\.?", re.I),
    re.compile(r"companies can search these words[^.]*\.?", re.I),
)


def strip_noise(text: str | None) -> str:
    """Remove URLs and known board furniture. For modelling input only.

    Deliberately not applied before storing: `postings.description` stays as
    the board wrote it, because the stored row is evidence and because the
    next piece of furniture we discover has to be removable from data already
    collected. This runs at feature time instead, where it is cheap to change.
    """
    if not text:
        return ""
    cleaned = URL.sub(" ", text)
    for pattern in _BOILERPLATE:
        cleaned = pattern.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


REMOTE_HINTS = re.compile(
    r"\b(?:remote|work\s+from\s+home|wfh|anywhere|distributed|telecommute)\b", re.I
)

# Boards write "Remote (US only)" and similar in the location field constantly.
_LOCATION_NOISE = re.compile(r"\b(?:remote|hybrid|on-?site|worldwide|anywhere)\b", re.I)

# The top strings actually seen in the data, mapped to something consistent.
# This is deliberately a lookup table and not a geocoder. Anything not in here
# is passed through as-is rather than mangled into a wrong city.
LOCATION_ALIASES = {
    "usa": "United States",
    "us": "United States",
    "u.s.": "United States",
    "united states of america": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "england": "United Kingdom",
    "nyc": "New York, United States",
    "new york city": "New York, United States",
    "sf": "San Francisco, United States",
    "sf bay area": "San Francisco, United States",
    "bay area": "San Francisco, United States",
    "bengaluru": "Bangalore, India",
    "blr": "Bangalore, India",
    "emea": "Europe",
    "apac": "Asia Pacific",
}


def is_remote(*fields: str | None) -> bool:
    """True if any of the given strings advertises remote work."""
    return any(REMOTE_HINTS.search(f) for f in fields if f)


def normalise_location(raw: str | None) -> str | None:
    """Best-effort tidy of a location string. Returns None for remote-only."""
    if not raw:
        return None
    text = _LOCATION_NOISE.sub(" ", raw)
    text = re.sub(r"[(){}\[\]]", " ", text)
    text = re.sub(r"\s*[/|,]\s*", ", ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,-")
    if not text:
        return None
    return LOCATION_ALIASES.get(text.lower(), text)


_PUNCT = re.compile(r"[^a-z0-9\s]")
_SENIORITY_NOISE = re.compile(
    r"\b(?:senior|junior|sr|jr|lead|staff|principal|i{1,3}|iv|v)\b"
)


def normalise_for_hash(value: str | None) -> str:
    if not value:
        return ""
    text = _PUNCT.sub(" ", value.lower())
    return re.sub(r"\s+", " ", text).strip()


def content_hash(title: str, company: str, location: str | None) -> str:
    """Fingerprint used to spot the same job posted to two boards.
    Exact match on normalised text only. This misses "Senior ML Engineer" vs
    "ML Engineer (Senior)" and we know it does. Phase 3 replaces this with
    embedding similarity, and the duplicate count from this version is the
    baseline that improvement gets measured against.
    """
    parts = [
        _SENIORITY_NOISE.sub(" ", normalise_for_hash(title)).strip(),
        normalise_for_hash(company),
        normalise_for_hash(location),
    ]
    joined = "|".join(re.sub(r"\s+", " ", p).strip() for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()
