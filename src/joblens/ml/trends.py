"""The numbers behind the dashboard. Plain pandas, no model.

Nothing here is clever, and that is deliberate: these are the aggregates a
user actually asks for ("what is in demand", "where are the jobs", "does
knowing Kubernetes pay"), and they should be readable by anyone checking
whether the dashboard is lying.

The one thing worth being careful about is the denominator. Every share is
reported against the postings that could have answered the question, not
against the whole corpus: salary figures only over postings with a parsed
salary, trends only over postings with a date. Mixing those two is how a
dashboard ends up claiming half the market is remote when really half the
postings did not say.
"""

from __future__ import annotations

import pandas as pd

from joblens.ml import dataset
from joblens.ml.salary_model import FX_TO_USD, MAX_ANNUAL_USD, MIN_ANNUAL_USD
from joblens.ml.skills import skill_counts, skill_matrix


def top_skills(frame: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Most requested skills, with the share of postings asking for each."""
    if frame.empty:
        return pd.DataFrame(columns=["skill", "postings", "share"])
    counts = skill_counts(frame).head(top_n)
    return pd.DataFrame(
        {
            "skill": counts.index,
            "postings": counts.to_numpy(),
            "share": counts.to_numpy() / len(frame),
        }
    )


def skill_trend(
    frame: pd.DataFrame,
    freq: str = "W",
    top_n: int = 8,
    skills: list[str] | None = None,
) -> pd.DataFrame:
    """Share of postings mentioning each top skill, per period.

    Share rather than count, because the number of postings collected per week
    depends on how the scraper ran that week. A count chart mostly plots our
    own uptime; a share chart plots the market.
    """
    dated = frame[frame["posted_at"].notna()]
    if dated.empty:
        return pd.DataFrame()
    chosen = skills or list(top_skills(dated, top_n)["skill"])
    if not chosen:
        return pd.DataFrame()
    matrix = skill_matrix(dated)[chosen].astype(float)
    matrix.index = pd.DatetimeIndex(dated["posted_at"])
    return matrix.resample(freq).mean().dropna(how="all")


def demand_by_location(frame: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Posting counts by coarse region, with the remote share alongside."""
    if frame.empty:
        return pd.DataFrame(columns=["region", "postings", "remote_share"])
    work = dataset.with_region(frame)
    grouped = (
        work.groupby("region")
        .agg(postings=("region", "size"), remote_share=("is_remote", "mean"))
        .sort_values("postings", ascending=False)
        .head(top_n)
        .reset_index()
    )
    return grouped


def remote_share(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    return float(frame["is_remote"].mean())


def salary_by_skill(
    frame: pd.DataFrame, min_postings: int = 5, top_n: int = 20
) -> pd.DataFrame:
    """Median advertised salary for postings mentioning each skill.

    This is an association and nothing more. Skills cluster with seniority and
    with industry, so "mlops pays more" here mostly means "postings that say
    mlops are also more senior". The regression in salary_model.py is the
    place that tries to hold the other columns still; this table does not.
    """
    priced = _priced(frame)
    if priced.empty:
        return pd.DataFrame(columns=["skill", "postings", "median_usd"])

    matrix = skill_matrix(priced)
    overall = float(priced["salary_usd"].median())
    rows = []
    for skill in matrix.columns:
        members = priced.loc[matrix[skill].to_numpy()]
        if len(members) < min_postings:
            continue
        median = float(members["salary_usd"].median())
        rows.append(
            {
                "skill": skill,
                "postings": len(members),
                "median_usd": median,
                "vs_overall": median / overall - 1,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["skill", "postings", "median_usd"])
    return (
        pd.DataFrame(rows)
        .sort_values("median_usd", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def _priced(frame: pd.DataFrame) -> pd.DataFrame:
    """Postings with a salary we believe, converted to USD."""
    if frame.empty:
        return frame
    work = frame.copy()
    midpoint = work[["salary_min_year", "salary_max_year"]].mean(axis=1, skipna=True)
    work["salary_usd"] = midpoint * work["salary_currency"].fillna("USD").map(FX_TO_USD)
    return work[work["salary_usd"].between(MIN_ANNUAL_USD, MAX_ANNUAL_USD)]


def summary(frame: pd.DataFrame, days: int = 90) -> dict:
    """One dict with everything the dashboard and the CLI both want."""
    if frame.empty:
        return {"postings": 0}

    recent = frame
    if frame["posted_at"].notna().any():
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        recent = frame[frame["posted_at"].isna() | (frame["posted_at"] >= cutoff)]

    priced = _priced(recent)
    return {
        "window_days": days,
        "postings": int(len(recent)),
        "sources": recent["source"].value_counts().to_dict(),
        "remote_share": round(remote_share(recent), 3),
        "with_salary": int(len(priced)),
        "salary_coverage": round(len(priced) / len(recent), 3) if len(recent) else 0.0,
        "median_salary_usd": (
            round(float(priced["salary_usd"].median())) if len(priced) else None
        ),
        "top_skills": top_skills(recent, 15).to_dict(orient="records"),
        "top_regions": demand_by_location(recent, 10).to_dict(orient="records"),
    }
