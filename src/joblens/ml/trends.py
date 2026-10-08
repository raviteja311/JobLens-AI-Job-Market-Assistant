"""The numbers behind the dashboard. Plain pandas, no model.

Nothing here is clever, and that is deliberate: these are the aggregates a
user actually asks for ("what is in demand", "where are the jobs"), and
they should be readable by anyone checking
whether the dashboard is lying.

The one thing worth being careful about is the denominator. Every share is
reported against the postings that could have answered the question, not
against the whole corpus: trends only over postings with a date. Mixing
those is how a
dashboard ends up claiming half the market is remote when really half the
postings did not say.
"""

from __future__ import annotations

import pandas as pd

from joblens.ml import dataset
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


def summary(frame: pd.DataFrame, days: int = 90) -> dict:
    """One dict with everything the dashboard and the CLI both want."""
    if frame.empty:
        return {"postings": 0}

    recent = frame
    if frame["posted_at"].notna().any():
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        recent = frame[frame["posted_at"].isna() | (frame["posted_at"] >= cutoff)]

    return {
        "window_days": days,
        "postings": int(len(recent)),
        "sources": recent["source"].value_counts().to_dict(),
        "remote_share": round(remote_share(recent), 3),
        "top_skills": top_skills(recent, 15).to_dict(orient="records"),
        "top_regions": demand_by_location(recent, 10).to_dict(orient="records"),
    }
