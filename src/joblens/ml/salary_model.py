"""Salary regression on the postings that published a number.

Order of business, and it is deliberate: predict the median, then a linear
model, then trees. Each one has to beat the one before it on the same folds or
it does not go in. A gradient booster that cannot beat `median` is not a
model, it is a slower way to be wrong, and knowing which of your models is in
that state is most of what this file is for.

Two things about this data that shape everything below:

  * Only a minority of postings state a salary at all, and the ones that do
    are not a random sample. Boards that require a salary field skew the set
    towards Adzuna and away from Hacker News. Every score here is therefore a
    score on salary-disclosing postings, not on the job market.
  * Salaries arrive in several currencies. They are converted with frozen
    rates below, which is wrong by however much the rates have moved. It is
    still better than comparing 90,000 INR against 90,000 USD, and it is
    listed in the README limitations rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from joblens.ml import dataset
from joblens.ml.skills import SkillEncoder, posting_text

RANDOM_STATE = 42

# Frozen on 2025-09-01. Refreshing them retroactively would change every past
# experiment's numbers, so they stay fixed and the date is the caveat.
FX_TO_USD = {
    "USD": 1.0,
    "GBP": 1.34,
    "EUR": 1.17,
    "CAD": 0.74,
    "AUD": 0.67,
    "INR": 0.0114,
    "JPY": 0.0068,
    "CHF": 1.25,
    "SEK": 0.096,
}

# Outside this band the row is a parser bug, not a salary: a stipend quoted
# per month read as annual, or a number that was really a job reference.
MIN_ANNUAL_USD = 10_000
MAX_ANNUAL_USD = 1_000_000

ARTIFACTS_DIR = Path(__file__).resolve().parents[3] / "artifacts"

CATEGORICAL = ["source", "region", "seniority", "is_remote"]


def training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for regression, with the target and features attached.

    Returns an empty frame rather than raising when nothing qualifies, so the
    caller can say "no salary data yet" instead of crashing on day one.
    """
    if frame.empty:
        return frame
    work = dataset.with_region(frame)
    work["seniority"] = work["title"].map(dataset.seniority)
    work["text"] = posting_text(work)

    midpoint = work[["salary_min_year", "salary_max_year"]].mean(axis=1, skipna=True)
    rate = work["salary_currency"].fillna("USD").map(FX_TO_USD)
    # An unrecognised currency code is dropped, not assumed to be dollars.
    work["salary_usd"] = midpoint * rate

    # Each stated bound has to be plausible, not only the midpoint: a range
    # misread as 100 to 200,000 has a believable midpoint and a nonsense floor.
    low = work["salary_min_year"] * rate
    high = work["salary_max_year"] * rate
    usable = (
        work["salary_usd"].between(MIN_ANNUAL_USD, MAX_ANNUAL_USD)
        & (low.isna() | (low >= MIN_ANNUAL_USD))
        & (high.isna() | (high <= MAX_ANNUAL_USD))
    )
    return work.loc[usable].reset_index(drop=True)


def _text_features() -> ColumnTransformer:
    """What the linear model sees: the words, plus the structured columns."""
    return ColumnTransformer(
        [
            (
                "text",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20_000,
                    sublinear_tf=True,
                ),
                "text",
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=2),
                CATEGORICAL,
            ),
        ]
    )


def _skill_features() -> ColumnTransformer:
    """What the tree models see: named skill flags instead of TF-IDF columns.

    Trees on a 20k-column sparse matrix are slow and their importances are
    unreadable. Feeding them the skill dictionary instead costs some accuracy
    and buys an answer to "which skills move the number", which is the whole
    reason anyone looks at this model.
    """
    return ColumnTransformer(
        [
            ("skills", SkillEncoder(), "text"),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=2),
                CATEGORICAL,
            ),
        ]
    )


def build_pipeline(name: str) -> Pipeline:
    """One named model. Everything is a Pipeline so CV cannot leak."""
    if name == "median":
        # No features on purpose: this is the number to beat.
        return Pipeline([("model", DummyRegressor(strategy="median"))])
    if name == "ridge":
        return Pipeline([("features", _text_features()), ("model", Ridge(alpha=1.0))])
    if name == "ridge_log":
        # Salaries are right-skewed, so squared error spends all its effort on
        # the top end. Training on log(salary) fixes that; the wrapper undoes
        # the transform so the reported MAE is still in dollars.
        return Pipeline(
            [
                ("features", _text_features()),
                (
                    "model",
                    TransformedTargetRegressor(
                        regressor=Ridge(alpha=1.0),
                        func=np.log1p,
                        inverse_func=np.expm1,
                    ),
                ),
            ]
        )
    if name == "random_forest":
        return Pipeline(
            [
                ("features", _skill_features()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if name == "gradient_boosting":
        return Pipeline(
            [
                ("features", _skill_features()),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    raise ValueError(f"unknown model {name!r}. known: {', '.join(MODELS)}")


MODELS = ("median", "ridge", "ridge_log", "random_forest", "gradient_boosting")


@dataclass(frozen=True)
class ModelScore:
    name: str
    mae: float
    rmse: float
    r2: float
    folds: int
    rows: int

    def beats(self, other: ModelScore) -> bool:
        return self.mae < other.mae


@dataclass(frozen=True)
class ComparisonReport:
    scores: list[ModelScore]

    @property
    def best(self) -> ModelScore:
        return min(self.scores, key=lambda score: score.mae)

    @property
    def baseline(self) -> ModelScore:
        return next(s for s in self.scores if s.name == "median")

    def as_table(self) -> str:
        """Markdown, because it gets pasted straight into the README."""
        head = self.scores[0]
        lines = [
            f"{head.rows} postings with a usable salary, "
            f"{head.folds}-fold cross-validation",
            "",
            "| model | MAE (USD) | RMSE (USD) | R2 | vs median |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        baseline_mae = self.baseline.mae
        for score in sorted(self.scores, key=lambda s: s.mae):
            delta = (baseline_mae - score.mae) / baseline_mae * 100
            lines.append(
                f"| {score.name} | {score.mae:,.0f} | {score.rmse:,.0f} "
                f"| {score.r2:.3f} | {delta:+.1f}% |"
            )
        return "\n".join(lines)


def compare_models(
    frame: pd.DataFrame, folds: int = 5, names: tuple[str, ...] = MODELS
) -> ComparisonReport:
    """Cross-validate every model on identical folds and rank by MAE.

    MAE and not RMSE as the headline: the errors that matter here are the
    everyday ones, and RMSE lets a handful of mislabelled 900k postings pick
    the winner. Both are reported so the gap between them stays visible.
    """
    dataset.require_rows(frame, 30, "salary regression")
    folds = max(2, min(folds, len(frame) // 5))
    splitter = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    X = frame[["text", *CATEGORICAL]]
    y = frame["salary_usd"].to_numpy()

    scores: list[ModelScore] = []
    for name in names:
        result = cross_validate(
            build_pipeline(name),
            X,
            y,
            cv=splitter,
            scoring=(
                "neg_mean_absolute_error",
                "neg_root_mean_squared_error",
                "r2",
            ),
            error_score="raise",
        )
        scores.append(
            ModelScore(
                name=name,
                mae=float(-result["test_neg_mean_absolute_error"].mean()),
                rmse=float(-result["test_neg_root_mean_squared_error"].mean()),
                r2=float(result["test_r2"].mean()),
                folds=folds,
                rows=len(frame),
            )
        )
    return ComparisonReport(scores)


def fit(frame: pd.DataFrame, name: str) -> Pipeline:
    pipeline = build_pipeline(name)
    pipeline.fit(frame[["text", *CATEGORICAL]], frame["salary_usd"].to_numpy())
    return pipeline


def fit_and_save(frame: pd.DataFrame, name: str, path: Path | None = None) -> Path:
    """Train on everything and write the artifact. Not tracked in git."""
    pipeline = fit(frame, name)
    path = path or ARTIFACTS_DIR / "salary_model.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"name": name, "pipeline": pipeline}, path)
    return path


def load_model(path: Path | None = None) -> tuple[str, Pipeline]:
    payload = joblib.load(path or ARTIFACTS_DIR / "salary_model.joblib")
    return payload["name"], payload["pipeline"]


def column_importance(
    frame: pd.DataFrame, name: str, folds: int = 5, n_repeats: int = 10
) -> pd.DataFrame:
    """Permutation importance of each input column, in dollars of MAE.

    `top_features` reads a fitted model's own weights, which answers "which
    TF-IDF term did the ridge lean on". This answers the question a person
    asks first: does the text matter at all, or is it the region and the
    seniority doing the work. Each column is shuffled on held-out rows and
    the rise in MAE is the column's importance; a column whose shuffle
    changes nothing was decoration. Model-agnostic, so it works for the
    trees as well as the ridge, and honest, because it is scored out of
    sample.

    Scored on every fold of the same K-fold split `compare_models` uses, not
    on one train/test split. With about 110 priced rows a 25% split holds 28
    of them, and the first version of this function, run on two corpora an
    ingest apart, ranked text first by 11.8k and then source first by 8.6k.
    `std_usd` is the spread across folds and repeats, which is the number
    that says whether a ranking is real.
    """
    dataset.require_rows(frame, 30, "permutation importance")
    columns = ["text", *CATEGORICAL]
    X = frame[columns].reset_index(drop=True)
    y = frame["salary_usd"].to_numpy()
    folds = max(2, min(folds, len(frame) // 5))
    splitter = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, test_idx in splitter.split(X):
        pipeline = build_pipeline(name).fit(X.iloc[train_idx], y[train_idx])
        result = permutation_importance(
            pipeline,
            X.iloc[test_idx],
            y[test_idx],
            scoring="neg_mean_absolute_error",
            n_repeats=n_repeats,
            random_state=RANDOM_STATE,
        )
        scores.append(result.importances)
    stacked = np.hstack(scores)
    table = pd.DataFrame(
        {
            "column": columns,
            "mae_increase_usd": stacked.mean(axis=1),
            "std_usd": stacked.std(axis=1),
        }
    )
    return table.sort_values("mae_increase_usd", ascending=False).reset_index(drop=True)


def top_features(pipeline: Pipeline, top_k: int = 20) -> pd.DataFrame:
    """Which inputs the fitted model leans on, largest effect first.

    Coefficients for the linear models, impurity importance for the trees.
    Neither is a causal claim: "aws appears in postings that pay more" is not
    "learning aws pays more", and the experiments log repeats that warning.
    """
    features = pipeline.named_steps.get("features")
    if features is None:
        return pd.DataFrame(columns=["feature", "weight"])
    names = np.asarray(features.get_feature_names_out(), dtype=object)

    model = pipeline.named_steps["model"]
    if isinstance(model, TransformedTargetRegressor):
        model = model.regressor_

    if hasattr(model, "coef_"):
        weights = np.ravel(model.coef_)
    elif hasattr(model, "feature_importances_"):
        weights = np.ravel(model.feature_importances_)
    else:
        return pd.DataFrame(columns=["feature", "weight"])

    order = np.argsort(np.abs(weights))[::-1][:top_k]
    return pd.DataFrame(
        {"feature": names[order], "weight": weights[order]}
    ).reset_index(drop=True)
