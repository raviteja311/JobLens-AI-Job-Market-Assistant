import pandas as pd

from joblens.ml import trends


def test_top_skills_shares_are_fractions(corpus):
    table = trends.top_skills(corpus, top_n=10)
    assert len(table) == 10
    assert table["share"].between(0, 1).all()
    assert table["postings"].is_monotonic_decreasing


def test_top_skills_of_an_empty_corpus_is_empty():
    empty = trends.top_skills(pd.DataFrame())
    assert empty.empty
    assert list(empty.columns) == ["skill", "postings", "share"]


def test_skill_trend_is_a_share_per_period(corpus):
    trend = trends.skill_trend(corpus, freq="W", top_n=5)
    assert not trend.empty
    assert trend.shape[1] == 5
    assert trend.to_numpy().max() <= 1.0
    assert isinstance(trend.index, pd.DatetimeIndex)


def test_skill_trend_without_dates_is_empty(corpus):
    undated = corpus.copy()
    undated["posted_at"] = pd.NaT
    assert trends.skill_trend(undated).empty


def test_demand_by_location_counts_and_remote_share(corpus):
    table = trends.demand_by_location(corpus, top_n=5)
    assert table["postings"].sum() <= len(corpus)
    assert table["remote_share"].between(0, 1).all()
    # Postings with no location land in one bucket rather than being dropped.
    assert "unknown" in set(table["region"])


def test_remote_share_is_a_fraction(corpus):
    assert 0.0 <= trends.remote_share(corpus) <= 1.0
    assert trends.remote_share(pd.DataFrame()) == 0.0


def test_salary_by_skill_only_reports_well_supported_skills(corpus):
    table = trends.salary_by_skill(corpus, min_postings=5)
    assert (table["postings"] >= 5).all()
    assert (table["median_usd"] > 0).all()
    assert table["median_usd"].is_monotonic_decreasing


def test_salary_by_skill_ranks_the_better_paid_family_higher(corpus):
    table = trends.salary_by_skill(corpus, min_postings=3, top_n=100).set_index("skill")
    # The synthetic LLM roles pay more than the analyst roles, and the skills
    # that only appear in each should come out in that order.
    assert table.loc["rag", "median_usd"] > table.loc["tableau", "median_usd"]


def test_summary_counts_its_own_denominators(corpus):
    result = trends.summary(corpus, days=365)
    assert result["postings"] == len(corpus)
    assert result["with_salary"] < result["postings"]
    assert result["salary_coverage"] == round(
        result["with_salary"] / result["postings"], 3
    )
    assert result["median_salary_usd"] > 0
    assert len(result["top_skills"]) == 15


def test_summary_of_an_empty_corpus_does_not_explode():
    assert trends.summary(pd.DataFrame()) == {"postings": 0}
