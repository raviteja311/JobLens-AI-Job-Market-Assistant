"""Getting postings out of Postgres and into a DataFrame.

One place that knows the column names, so a schema change breaks here and
nowhere else. Every function below also works on a frame built by hand, which
is what the tests use: none of the modelling code needs a database.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from joblens import db
from joblens.models import Posting

COLUMNS = (
    "id",
    "source",
    "source_id",
    "title",
    "company",
    "location",
    "is_remote",
    "salary_raw",
    "salary_min_year",
    "salary_max_year",
    "salary_currency",
    "description",
    "url",
    "posted_at",
    "content_hash",
    "first_seen_at",
)

SELECT_SQL = f"select {', '.join(COLUMNS)} from postings"  # noqa: S608 - fixed list


def load_postings(source: str | None = None, conn=None) -> pd.DataFrame:
    """Every posting in the silver table, optionally for one source."""
    if conn is not None:
        return _query(conn, source)
    with db.connect() as connection:
        return _query(connection, source)


def _query(conn, source: str | None) -> pd.DataFrame:
    if source:
        rows = conn.execute(SELECT_SQL + " where source = %s", (source,)).fetchall()
    else:
        rows = conn.execute(SELECT_SQL).fetchall()
    return _normalise(pd.DataFrame(rows, columns=list(COLUMNS)))


def frame_from_postings(postings: Iterable[Posting]) -> pd.DataFrame:
    """Build the same frame from Posting objects, without touching the DB."""
    rows = [posting.model_dump() for posting in postings]
    frame = pd.DataFrame(rows)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return _normalise(frame[list(COLUMNS)])


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Force the dtypes the models assume, whatever the source gave us."""
    if frame.empty:
        return frame
    frame = frame.copy()
    for column in ("salary_min_year", "salary_max_year"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("posted_at", "first_seen_at"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    frame["is_remote"] = frame["is_remote"].fillna(False).astype(bool)
    for column in ("title", "company", "description", "source"):
        frame[column] = frame[column].fillna("").astype(str)
    return frame


# Everything after the last comma, which for these boards is almost always the
# country or the US state. Good enough to group by; not a geocoder, and the
# clustering and trend code says so where it matters.
_TRAILING_PART = re.compile(r"[^,]+$")


def region(location: str | None) -> str:
    """Coarse grouping key for a location string. 'unknown' when there is none.

    The pd.isna check is not decoration: a missing location arrives as NaN,
    NaN is truthy, and str(NaN) is "nan", so without it the dashboard grows a
    region called "Nan" that outranks most real countries.
    """
    if location is None or pd.isna(location) or not str(location).strip():
        return "unknown"
    match = _TRAILING_PART.search(str(location).strip())
    if not match:
        return "unknown"
    part = match.group(0).strip()
    # Title-casing everything turns "CA" into "Ca" and "UK" into "Uk", which
    # looks like a bug on the dashboard because it is one. Short all-letter
    # parts are state or country codes and keep their case.
    return part.upper() if len(part) <= 3 and part.isalpha() else part.title()


def with_region(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["region"] = frame["location"].map(region)
    return frame


_SENIOR = re.compile(r"\b(?:senior|sr|lead|staff|principal|head|director)\b", re.I)
_JUNIOR = re.compile(
    r"\b(?:junior|jr|intern|internship|graduate|entry[- ]level)\b", re.I
)


def seniority(title: str | None) -> str:
    """Three buckets from the title. Postings that say nothing are 'mid'."""
    text = title or ""
    if _JUNIOR.search(text):
        return "junior"
    if _SENIOR.search(text):
        return "senior"
    return "mid"


def load_embeddings(
    frame: pd.DataFrame,
    conn=None,
    strategy: str = "whole",
    model: str = "all-MiniLM-L6-v2",
) -> tuple[pd.DataFrame, np.ndarray]:
    """The stored vector for each posting in `frame`, from the Phase 3 index.

    Returns the subset of the frame that has a vector, in the same order as
    the matrix rows. Postings that were never embedded are dropped rather
    than zero-filled, because a zero vector is a real point to k-means.
    """
    if frame.empty:
        return frame, np.empty((0, 0), dtype=np.float32)
    if conn is None:
        with db.connect() as connection:
            return load_embeddings(frame, connection, strategy, model)
    rows = conn.execute(
        """
        select posting_id, embedding::text as embedding
          from posting_chunks
         where strategy = %s and model = %s and chunk_index = 0
           and posting_id = any(%s)
        """,
        (strategy, model, [int(i) for i in frame["id"]]),
    ).fetchall()
    vectors = {
        # pgvector's text form is a JSON array.
        r["posting_id"]: np.asarray(json.loads(r["embedding"]), dtype=np.float32)
        for r in rows
    }
    keep = frame["id"].map(lambda i: int(i) in vectors)
    subset = frame.loc[keep].reset_index(drop=True)
    matrix = (
        np.vstack([vectors[int(i)] for i in subset["id"]])
        if len(subset)
        else np.empty((0, 0), dtype=np.float32)
    )
    return subset, matrix


def require_rows(frame: pd.DataFrame, minimum: int, what: str) -> None:
    """Fail with a useful message instead of a shape error 40 lines later."""
    if len(frame) < minimum:
        raise ValueError(
            f"{what} needs at least {minimum} rows, got {len(frame)}. "
            "Run `python -m joblens ingest` for a few more days first."
        )
