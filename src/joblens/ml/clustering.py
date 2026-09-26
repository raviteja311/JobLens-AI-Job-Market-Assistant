"""Unsupervised grouping of postings: TF-IDF, then k-means.

Job titles are useless as categories. "ML Engineer", "Applied Scientist",
"Member of Technical Staff" and "AI Engineer" describe the same work, and one
company's "Data Scientist" is another's "Analyst". Clustering gives the
dashboard groupings that come from what the posting says rather than what the
company decided to call it.

k-means runs on the TF-IDF matrix directly rather than on a reduced space, and
that is the point: the cluster centres stay in vocabulary space, so each
cluster can be labelled with the words that define it. A cluster you cannot
name is a cluster you cannot put on a dashboard.

The same routine also runs on the Phase 3 embeddings. The clusters are then
found in embedding space but still labelled from TF-IDF, by averaging the
TF-IDF rows of each cluster's members, so the two representations can be
compared on the one thing that matters for a dashboard: whether the groups
have names a person would recognise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score

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
    representation: str = "tfidf"

    def as_table(self) -> str:
        lines = [
            f"{self.representation}: k = {self.k}, silhouette {self.silhouette:.3f}, "
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
    frame: pd.DataFrame,
    k: int | None = None,
    terms_per_cluster: int = 10,
    vectors: np.ndarray | None = None,
) -> ClusterResult:
    """Cluster the corpus and label each group. k defaults to best silhouette.

    With `vectors` (one embedding per row of `frame`) k-means runs in
    embedding space; without them it runs on TF-IDF. Labels always come from
    TF-IDF, from the mean of each cluster's rows, which for the TF-IDF run
    is exactly the k-means centroid and for the embedding run is the nearest
    thing to one that has words attached.
    """
    dataset.require_rows(frame, 20, "clustering")
    vectoriser, tfidf = _vectorise(frame)
    if vectors is not None:
        if len(vectors) != len(frame):
            raise ValueError(
                f"{len(vectors)} vectors for {len(frame)} postings; pass the frame "
                "returned by dataset.load_embeddings alongside its matrix"
            )
        space = np.asarray(vectors, dtype=np.float64)
        representation = "embedding"
    else:
        space = tfidf
        representation = "tfidf"

    candidates = pd.DataFrame(columns=["k", "silhouette"])
    if k is None:
        candidates = choose_k(space)
        if candidates.empty:
            raise ValueError("no usable k: the corpus is too small or too uniform")
        k = int(candidates.loc[candidates["silhouette"].idxmax(), "k"])

    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = model.fit_predict(space)
    score = (
        float(silhouette_score(space, labels)) if len(set(labels)) > 1 else float("nan")
    )

    vocabulary = np.asarray(vectoriser.get_feature_names_out(), dtype=object)
    texts = posting_text(frame).to_numpy()

    rows = []
    for cluster in range(k):
        members = labels == cluster
        centre = np.asarray(tfidf[members].mean(axis=0)).ravel()
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
        representation=representation,
    )


@dataclass(frozen=True)
class RepresentationComparison:
    """TF-IDF and embedding clusterings of the same postings, side by side.

    Silhouettes are not comparable across the two spaces: each is measured
    in its own geometry. The adjusted Rand index is the number that compares
    them, because it only asks whether the two partitions agree about which
    postings belong together. 1.0 is identical, 0.0 is chance.
    """

    tfidf: ClusterResult
    embedding: ClusterResult
    agreement: float

    def as_table(self) -> str:
        lines = [
            f"{len(self.tfidf.labels)} postings, adjusted Rand index "
            f"{self.agreement:.3f} between the two partitions",
            "",
            "| representation | k | silhouette | clusters (postings) |",
            "| --- | ---: | ---: | --- |",
        ]
        for result in (self.tfidf, self.embedding):
            named = ", ".join(
                f"{row.label} ({row.postings})" for row in result.summary.itertuples()
            )
            lines.append(
                f"| {result.representation} | {result.k} | {result.silhouette:.3f} "
                f"| {named} |"
            )
        return "\n".join(lines)


def compare_representations(
    frame: pd.DataFrame, vectors: np.ndarray, k: int | None = None
) -> RepresentationComparison:
    """Cluster the same postings in both spaces and measure the agreement."""
    tfidf = cluster_postings(frame, k=k)
    embedding = cluster_postings(frame, k=k, vectors=vectors)
    return RepresentationComparison(
        tfidf=tfidf,
        embedding=embedding,
        agreement=float(adjusted_rand_score(tfidf.labels, embedding.labels)),
    )
