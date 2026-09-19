"""Two chunking strategies, so the choice can be measured instead of guessed.

A job posting is short by retrieval standards: the median description in this
corpus is around 1,500 characters, which fits inside a MiniLM context window
whole. So the obvious strategy is to not chunk at all.

The argument against it is that a posting is really several documents stapled
together: what the company does, what the role is, what they want from you,
and what they pay. Embedding all of that into one vector averages them, and a
query about one section gets diluted by the other three.

Both are implemented. `docs/experiments.md` has the numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from joblens.cleaning import strip_noise

# Roughly the model's window in characters. MiniLM truncates at 256 word
# pieces; this keeps a chunk under that without needing a tokeniser here.
MAX_CHARS = 900
MIN_CHARS = 120


@dataclass(frozen=True)
class Chunk:
    posting_id: int
    strategy: str
    index: int
    content: str


def _header(title: str, company: str) -> str:
    # Not `f"{title} at {company}".strip(" at")`: str.strip takes a set of
    # characters, so that turns "Data Analyst at " into "Data Analys".
    title = (title or "").strip()
    company = (company or "").strip()
    return f"{title} at {company}" if company else title


def chunk_whole(
    posting_id: int, title: str, company: str, description: str
) -> list[Chunk]:
    """One vector for the whole posting. The baseline."""
    body = strip_noise(description)
    content = f"{_header(title, company)}. {body}".strip()
    return [Chunk(posting_id, "whole", 0, content[: MAX_CHARS * 2])]


# Blank lines are what job boards actually use to separate sections. Headings
# ("Requirements:", "What you'll do") are a bonus when they exist and absent
# from most Hacker News comments, so they cannot be the primary split.
_PARAGRAPH = re.compile(r"\n\s*\n")


def chunk_sections(
    posting_id: int, title: str, company: str, description: str
) -> list[Chunk]:
    """Split on paragraphs, then pack into chunks under the model's window.

    Every chunk is prefixed with the title and company. Without that a chunk
    saying "5 years of Python" retrieves for any query mentioning Python and
    the caller has no idea which job it came from; the prefix costs a few
    tokens and makes each chunk independently meaningful.
    """
    body = strip_noise(description)
    header = _header(title, company)
    if not body:
        return [Chunk(posting_id, "section", 0, header)]

    chunks: list[str] = []
    current = ""
    for paragraph in _PARAGRAPH.split(body):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(current) + len(paragraph) + 1 <= MAX_CHARS:
            current = f"{current} {paragraph}".strip()
        else:
            if current:
                chunks.append(current)
            # A single paragraph longer than the window gets hard-split. Rare,
            # and losing a sentence boundary beats silently truncating.
            while len(paragraph) > MAX_CHARS:
                chunks.append(paragraph[:MAX_CHARS])
                paragraph = paragraph[MAX_CHARS:]
            current = paragraph
    if current:
        chunks.append(current)

    # A trailing scrap ("Apply by Friday.") is not worth its own vector.
    # Pop into a name first: `chunks[-2] = f"{chunks[-2]} {chunks.pop()}"`
    # evaluates the pop before resolving the target index, so a two-element
    # list becomes a one-element list and then assigns to [-2].
    if len(chunks) > 1 and len(chunks[-1]) < MIN_CHARS:
        tail = chunks.pop()
        chunks[-1] = f"{chunks[-1]} {tail}"

    return [
        Chunk(posting_id, "section", i, f"{header}. {text}")
        for i, text in enumerate(chunks)
    ]


STRATEGIES = {"whole": chunk_whole, "section": chunk_sections}


def chunk(strategy: str, posting_id: int, title: str, company: str, description: str):
    try:
        return STRATEGIES[strategy](posting_id, title, company, description)
    except KeyError:
        known = ", ".join(sorted(STRATEGIES))
        raise ValueError(
            f"unknown chunking strategy {strategy!r}. known: {known}"
        ) from None
