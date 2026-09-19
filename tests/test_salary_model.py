import pandas as pd
import pytest

from joblens.ml import salary_model


@pytest.fixture(scope="module")
def priced(corpus):
    return salary_model.training_frame(corpus)


def test_training_frame_keeps_only_postings_with_a_salary(priced, corpus):
    assert 0 < len(priced) < len(corpus)
    assert priced["salary_usd"].notna().all()
    assert "text" in priced.columns and "region" in priced.columns


def test_training_frame_converts_currency():
    frame = pd.DataFrame(
        {
            "title": ["ML Engineer"] * 2,
            "description": ["PyTorch"] * 2,
            "location": ["London, United Kingdom", "New York, United States"],
            "is_remote": [False, False],
            "source": ["adzuna", "adzuna"],
            "salary_min_year": [100_000, 100_000],
            "salary_max_year": [100_000, 100_000],
            "salary_currency": ["GBP", "USD"],
        }
    )
    result = salary_model.training_frame(frame)
    gbp, usd = result["salary_usd"].tolist()
    assert gbp > usd
    assert gbp == pytest.approx(100_000 * salary_model.FX_TO_USD["GBP"])


def test_absurd_salaries_are_dropped():
    frame = pd.DataFrame(
        {
            "title": ["Intern", "Founder"],
            "description": ["Python", "Python"],
            "location": [None, None],
            "is_remote": [True, True],
            "source": ["hackernews", "hackernews"],
            # A monthly stipend misread as annual, and a phone number.
            "salary_min_year": [1_200, 9_000_000],
            "salary_max_year": [1_200, 9_000_000],
            "salary_currency": ["USD", "USD"],
        }
    )
    assert salary_model.training_frame(frame).empty


def test_unknown_currency_is_dropped_not_assumed_to_be_dollars():
    frame = pd.DataFrame(
        {
            "title": ["ML Engineer"],
            "description": ["Python"],
            "location": [None],
            "is_remote": [True],
            "source": ["adzuna"],
            "salary_min_year": [100_000],
            "salary_max_year": [100_000],
            "salary_currency": ["ZWL"],
        }
    )
    assert salary_model.training_frame(frame).empty


def test_too_little_data_fails_loudly(corpus):
    small = salary_model.training_frame(corpus).head(5)
    with pytest.raises(ValueError, match="at least 30 rows"):
        salary_model.compare_models(small)


@pytest.mark.slow
def test_a_real_model_beats_predicting_the_median(priced):
    report = salary_model.compare_models(priced, folds=4)
    assert report.baseline.name == "median"
    assert report.best.name != "median"
    assert report.best.beats(report.baseline)
    assert report.best.r2 > 0.3


@pytest.mark.slow
def test_every_model_gets_scored_on_the_same_folds(priced):
    report = salary_model.compare_models(priced, folds=4)
    assert {score.name for score in report.scores} == set(salary_model.MODELS)
    assert len({score.folds for score in report.scores}) == 1
    assert len({score.rows for score in report.scores}) == 1


@pytest.mark.slow
def test_table_is_markdown_and_names_the_winner(priced):
    table = salary_model.compare_models(priced, folds=4).as_table()
    assert "| model | MAE (USD) |" in table
    assert "median" in table


@pytest.mark.slow
def test_tree_importances_are_readable_feature_names(priced):
    pipeline = salary_model.fit(priced, "random_forest")
    top = salary_model.top_features(pipeline, top_k=10)
    assert len(top) == 10
    # ColumnTransformer prefixes with the block name, so "skills__skill:aws".
    assert any("skill:" in name for name in top["feature"])
    assert any("seniority" in name for name in top["feature"])


@pytest.mark.slow
def test_saved_model_round_trips(priced, tmp_path):
    path = salary_model.fit_and_save(priced, "ridge", tmp_path / "model.joblib")
    name, pipeline = salary_model.load_model(path)
    assert name == "ridge"
    predicted = pipeline.predict(priced[["text", *salary_model.CATEGORICAL]].head(3))
    assert len(predicted) == 3
    assert (predicted > 0).all()
