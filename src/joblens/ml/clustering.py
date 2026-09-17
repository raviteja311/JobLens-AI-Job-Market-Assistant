"""Unsupervised grouping of postings: TF-IDF, then k-means.

Job titles are useless as categories. "ML Engineer", "Applied Scientist",
"Member of Technical Staff" and "AI Engineer" describe the same work, and one
company's "Data Scientist" is another's "Analyst". Clustering gives the
dashboard groupings that come from what the posting says rather than what the
company decided to call it.

k-means runs on the TF-IDF matrix directly rather than on a reduced space, and
that is the point: the cluster centres stay in vocabulary space, so each
cluster can be labelled with the words that define it. A cluster you cannot
name is a cluster you cannot put on a dashboard. Phase 3 re-runs this on
embeddings and the labels from this version are what the comparison is
against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from joblens.ml import dataset
from joblens.ml.skills import extract_skills, posting_text

RANDOM_STATE = 42
DEFAULT_K_RANGE = range(3, 11)


@dataclass(frozen=True)
class ClusterResult:
    k: int
    silhouette: float
    labels: np.ndarray
    summary: pd.DataFrame
    candidates: pd.DataFrame

    def as_table(self) -> str:
        lines = [
            f"k = {self.k}, silhouette {self.silhouette:.3f}, "
            f"{len(self.labels)} postings",
            "",
            "| cluster | postings | label | top terms | top skills |",
            "| ---: | ---: | --- | --- | --- |",
        ]
        for row in self.summary.itertuples():
            lines.append(
                f"| {row.cluster} | {row.postings} | {row.label} "
                f"| {', '.join(row.top_terms[:6])} "
                f"| {', '.join(row.top_skills[:5]) or '-'} |"
            )
        return "\n".join(lines)


def _vectorise(frame: pd.DataFrame) -> tuple[TfidfVectorizer, "np.ndarray"]:
    vectoriser = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.6,  # drop boilerplate: every posting says "team" and "experience"
        max_features=20_000,
        sublinear_tf=True,
    )
    return vectoriser, vectoriser.fit_transform(posting_text(frame))


def choose_k(matrix, k_range=DEFAULT_K_RANGE) -> pd.DataFrame:
    """Silhouette for each candidate k. Higher is better, 0 means no structure.

    Silhouette on sparse text is a weak signal, not an oracle: values in the
    0.01-0.05 band are normal here and the differences between them are small.
    It is used to avoid an arbitrary k, not to claim the clusters are crisp.
    """
    rows = []
    usable = [k for k in k_range if k < matrix.shape[0]]
    for k in usable:
        labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(
            matrix
        )
        if len(set(labels)) < 2:
            continue
        rows.append({"k": k, "silhouette": float(silhouette_score(matrix, labels))})
    return pd.DataFrame(rows)


def _label(terms: list[str], skills: list[str]) -> str:
    """A short human name for a cluster: its two strongest distinct terms."""
    parts: list[str] = []
    for term in terms:
        if not any(term in existing or existing in term for existing in parts):
            parts.append(term)
        if len(parts) == 2:
            break
    if skills and len(parts) < 2:
        parts.append(skills[0])
    return " / ".join(parts) if parts else "unlabelled"


def cluster_postings(
    frame: pd.DataFrame, k: int | None = None, terms_per_cluster: int = 10
) -> ClusterResult:
    """Cluster the corpus and label each group. k defaults to best silhouette."""
    dataset.require_rows(frame, 20, "clustering")
    vectoriser, matrix = _vectorise(frame)

    candidates = pd.DataFrame(columns=["k", "silhouette"])
    if k is None:
        candidates = choose_k(matrix)
        if candidates.empty:
            raise ValueError("no usable k: the corpus is too small or too uniform")
        k = int(candidates.loc[candidates["silhouette"].idxmax(), "k"])

    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = model.fit_predict(matrix)
    score = (
        float(silhouette_score(matrix, labels))
        if len(set(labels)) > 1
        else float("nan")
    )

    vocabulary = np.asarray(vectoriser.get_feature_names_out(), dtype=object)
    texts = posting_text(frame).to_numpy()

    rows = []
    for cluster in range(k):
        members = labels == cluster
        centre = model.cluster_centers_[cluster]
        top = np.argsort(centre)[::-1][:terms_per_cluster]
        terms = [str(term) for term in vocabulary[top]]

        counts: dict[str, int] = {}
        for text in texts[members]:
            for skill in extract_skills(text):
                counts[skill] = counts.get(skill, 0) + 1
        skills = sorted(counts, key=lambda s: counts[s], reverse=True)[:8]

        rows.append(
            {
                "cluster": cluster,
                "postings": int(members.sum()),
                "label": _label(terms, skills),
                "top_terms": terms,
                "top_skills": skills,
            }
        )

    summary = pd.DataFrame(rows).sort_values("postings", ascending=False)
    return ClusterResult(
        k=k,
        silhouette=score,
        labels=labels,
        summary=summary.reset_index(drop=True),
        candidates=candidates,
    )
