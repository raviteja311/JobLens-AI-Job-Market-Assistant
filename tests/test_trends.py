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


def test_summary_counts_its_own_denominators(corpus):
    result = trends.summary(corpus, days=365)
    assert result["postings"] == len(corpus)
    assert len(result["top_skills"]) == 15


def test_summary_of_an_empty_corpus_does_not_explode():
    assert trends.summary(pd.DataFrame()) == {"postings": 0}
