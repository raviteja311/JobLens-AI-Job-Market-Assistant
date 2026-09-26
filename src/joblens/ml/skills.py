"""Skill extraction, version 1: a dictionary and a regex.

This is deliberately the dumbest thing that works, and it is here for two
reasons. It is good enough to power the trends dashboard today, and it is the
number Phase 6's fine-tuned extractor has to beat. "We replaced an API call
with a 3B model" means nothing without knowing what a lookup table already
achieved.

The taxonomy is hand-written from the postings actually collected rather than
copied from a generic list, so it covers what this corpus contains and misses
everything else. `coverage()` reports how often it finds nothing at all,
which is the honest measure of how much the table is missing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from joblens.cleaning import strip_noise

# canonical name -> the strings that mean it in a job posting.
# Aliases are lowercase; matching is case-insensitive.
SKILL_ALIASES: dict[str, list[str]] = {
    # languages
    "python": ["python"],
    "r": ["r language", "r programming"],
    "java": ["java"],
    "scala": ["scala"],
    "go": ["golang", "go lang"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "csharp"],
    "javascript": ["javascript"],
    "typescript": ["typescript"],
    "rust": ["rust"],
    "bash": ["bash", "shell scripting"],
    # data stores and query
    "sql": ["sql"],
    "postgresql": ["postgres", "postgresql", "pgvector"],
    "mysql": ["mysql"],
    "mongodb": ["mongodb", "mongo"],
    "redis": ["redis"],
    "elasticsearch": ["elasticsearch", "opensearch"],
    "snowflake": ["snowflake"],
    "bigquery": ["bigquery"],
    "redshift": ["redshift"],
    # data tooling
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "spark": ["spark", "pyspark"],
    "hadoop": ["hadoop"],
    "kafka": ["kafka"],
    "airflow": ["airflow"],
    "dbt": ["dbt"],
    "databricks": ["databricks"],
    # classic ml
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "xgboost": ["xgboost"],
    "lightgbm": ["lightgbm"],
    "machine learning": ["machine learning"],
    "deep learning": ["deep learning"],
    "statistics": ["statistics", "statistical modelling", "statistical modeling"],
    "a/b testing": ["a/b testing", "ab testing", "experimentation"],
    "time series": ["time series", "forecasting"],
    "recommender systems": ["recommender", "recommendation systems"],
    "reinforcement learning": ["reinforcement learning", "rlhf"],
    # deep learning stack
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow"],
    "keras": ["keras"],
    "jax": ["jax"],
    "hugging face": ["hugging face", "huggingface"],
    "transformers": ["transformers", "transformer models"],
    # genai
    "llm": ["llm", "llms", "large language model", "large language models"],
    "rag": ["rag", "retrieval augmented generation"],
    "prompt engineering": ["prompt engineering", "prompting"],
    "fine-tuning": ["fine-tuning", "fine tuning", "finetuning"],
    "lora": ["lora", "peft", "qlora"],
    "embeddings": ["embeddings", "embedding models"],
    "vector database": [
        "vector database",
        "vector db",
        "pinecone",
        "weaviate",
        "faiss",
    ],
    "langchain": ["langchain"],
    "llamaindex": ["llamaindex", "llama index"],
    "nlp": ["nlp", "natural language processing"],
    "computer vision": ["computer vision", "image recognition"],
    # platform
    "aws": ["aws", "amazon web services"],
    "gcp": ["gcp", "google cloud"],
    "azure": ["azure"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "terraform": ["terraform"],
    "ci/cd": ["ci/cd", "cicd", "continuous integration"],
    "git": ["git", "github", "gitlab"],
    "linux": ["linux", "unix"],
    # serving and mlops
    "mlops": ["mlops"],
    "mlflow": ["mlflow"],
    "kubeflow": ["kubeflow"],
    "sagemaker": ["sagemaker"],
    "vertex ai": ["vertex ai"],
    "vllm": ["vllm"],
    "fastapi": ["fastapi"],
    "flask": ["flask"],
    "django": ["django"],
    "rest api": ["rest api", "restful"],
    "graphql": ["graphql"],
    "streamlit": ["streamlit"],
    # analysis and reporting
    "tableau": ["tableau"],
    "power bi": ["power bi", "powerbi"],
    "looker": ["looker"],
    "excel": ["excel"],
}

SKILLS: tuple[str, ...] = tuple(sorted(SKILL_ALIASES))

# \b is useless next to '+' and '#', so "c++" and "c#" need their own
# boundaries: any character that could continue a technology name blocks the
# match. Without this, "excel" matches inside "excellent".
#
# The dot needs care. It continues a name in "node.js" and "asp.net", and ends
# a sentence in "...and AWS." Treating it as a name character in both places
# is a real bug we shipped once: every skill that happened to fall at the end
# of a sentence went missing. So a dot only blocks when a letter or digit
# follows it.
_LEFT = r"(?<![a-z0-9+#])(?<![a-z0-9]\.)"
_RIGHT = r"(?![a-z0-9+#])(?!\.[a-z0-9])"


def _compile(aliases: Sequence[str]) -> re.Pattern[str]:
    parts = [_LEFT + re.escape(alias) + _RIGHT for alias in aliases]
    return re.compile("|".join(parts), re.IGNORECASE)


_PATTERNS: dict[str, re.Pattern[str]] = {
    skill: _compile(aliases) for skill, aliases in SKILL_ALIASES.items()
}


def extract_skills(text: str | None) -> list[str]:
    """Canonical skill names found in one posting, alphabetically."""
    if not text:
        return []
    return [skill for skill in SKILLS if _PATTERNS[skill].search(text)]


def posting_text(frame: pd.DataFrame) -> pd.Series:
    """Title plus description, which is what every model here reads.

    The title is repeated because it carries far more signal per word than the
    description does, and repetition is the cheapest way to weight it without
    maintaining a second vectoriser.

    The description goes through `strip_noise` first. Skip that and the models
    spend their capacity on URLs and per-board boilerplate; see the clustering
    entry in docs/experiments.md for what that looked like. The title gets the
    same treatment: Hacker News posters sometimes put their URL where the
    company name goes, and 18 of those were enough to make `https / www` a
    cluster of its own.
    """
    title = frame["title"].fillna("").astype(str).map(strip_noise)
    if "description" in frame.columns:
        description = frame["description"].fillna("").astype(str).map(strip_noise)
    else:
        description = pd.Series([""] * len(frame), index=frame.index)
    return (title + ". " + title + ". " + description).str.strip()


def skill_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per known skill, one row per posting."""
    found = posting_text(frame).map(lambda text: set(extract_skills(text)))
    data = {skill: [skill in hits for hits in found] for skill in SKILLS}
    return pd.DataFrame(data, index=frame.index)


def skill_counts(frame: pd.DataFrame) -> pd.Series:
    """How many postings mention each skill, most common first."""
    if frame.empty:
        return pd.Series(dtype=int)
    counts = skill_matrix(frame).sum().astype(int)
    return counts[counts > 0].sort_values(ascending=False)


def coverage(frame: pd.DataFrame) -> float:
    """Share of postings where the table found at least one skill.

    This is the honest score for a lookup table: everything it does not know
    about shows up here as a posting with zero hits. Phase 6 quotes this
    number as the thing the fine-tuned extractor had to improve on.
    """
    if frame.empty:
        return 0.0
    return float(skill_matrix(frame).any(axis=1).mean())


def tfidf_keywords(
    texts: Iterable[str], top_k: int = 10, max_features: int = 5000
) -> list[list[str]]:
    """Unsupervised keyword extraction, the second half of the baseline.

    The dictionary only knows what we told it. TF-IDF knows nothing but finds
    whatever is unusually frequent in a posting, so the two disagree in useful
    ways: terms ranked high here that are missing from SKILL_ALIASES are the
    queue of things to add to the table.
    """
    corpus = [text or "" for text in texts]
    if not corpus:
        return []
    vectoriser = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vectoriser.fit_transform(corpus)
    vocabulary = np.asarray(vectoriser.get_feature_names_out())
    keywords: list[list[str]] = []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        columns, scores = matrix.indices[start:end], matrix.data[start:end]
        order = np.argsort(scores)[::-1][:top_k]
        keywords.append([str(term) for term in vocabulary[columns[order]]])
    return keywords


class SkillEncoder(BaseEstimator, TransformerMixin):
    """sklearn adapter: free text in, one binary column per skill out.

    This is what lets the tree models train on named features instead of
    TF-IDF columns, and it is the only reason their feature importances are
    readable in the experiments log.
    """

    def fit(self, X, y=None):  # noqa: N803 - sklearn's argument name
        return self

    def transform(self, X):  # noqa: N803 - sklearn's argument name
        texts = pd.Series(np.asarray(X, dtype=object).ravel()).fillna("").astype(str)
        rows = [set(extract_skills(text)) for text in texts]
        return np.array(
            [[float(skill in hits) for skill in SKILLS] for hits in rows],
            dtype=float,
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array([f"skill:{skill}" for skill in SKILLS], dtype=object)
