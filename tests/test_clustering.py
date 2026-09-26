import numpy as np
import pandas as pd
import pytest

from joblens.ml import clustering


def test_too_little_data_fails_loudly():
    frame = pd.DataFrame({"title": ["ML Engineer"], "description": ["PyTorch"]})
    with pytest.raises(ValueError, match="at least 20 rows"):
        clustering.cluster_postings(frame)


@pytest.mark.slow
def test_finds_the_families_it_was_given(corpus):
    # The corpus has four job families with distinct vocabularies, so k=4
    # should separate them rather than producing one giant cluster.
    result = clustering.cluster_postings(corpus, k=4)
    assert result.k == 4
    assert len(result.labels) == len(corpus)
    assert result.summary["postings"].sum() == len(corpus)
    assert result.summary["postings"].min() > len(corpus) * 0.1


@pytest.mark.slow
def test_every_cluster_gets_a_label_and_terms(corpus):
    result = clustering.cluster_postings(corpus, k=4)
    assert result.summary["label"].ne("unlabelled").all()
    assert result.summary["top_terms"].map(len).min() > 0
    assert result.summary["top_skills"].map(len).min() > 0


@pytest.mark.slow
def test_clusters_are_named_after_their_own_vocabulary(corpus):
    result = clustering.cluster_postings(corpus, k=4)
    labels = " ".join(result.summary["label"])
    # Each family's signature tool should turn up in some cluster's label or
    # its top terms. Labels come from centroids, so this is a real check that
    # the grouping followed the text.
    terms = (
        labels
        + " "
        + " ".join(term for row in result.summary["top_terms"] for term in row)
    )
    assert "tableau" in terms
    assert "pytorch" in terms or "tensorflow" in terms


@pytest.mark.slow
def test_choosing_k_reports_a_score_per_candidate(corpus):
    result = clustering.cluster_postings(corpus)
    assert not result.candidates.empty
    assert set(result.candidates.columns) == {"k", "silhouette"}
    assert result.k == int(
        result.candidates.loc[result.candidates["silhouette"].idxmax(), "k"]
    )


@pytest.mark.slow
def test_table_renders(corpus):
    table = clustering.cluster_postings(corpus, k=3).as_table()
    assert "| cluster | postings | label |" in table
    assert "silhouette" in table


def _family_vectors(corpus):
    """A fake embedding that encodes the job family, so the embedding run has
    structure to find without loading a model."""
    families = sorted(
        corpus["title"].str.replace(r"^(Junior|Senior) ", "", regex=True).unique()
    )
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(len(families), 16))
    rows = []
    for title in corpus["title"]:
        family = title.replace("Junior ", "").replace("Senior ", "")
        rows.append(centres[families.index(family)] + rng.normal(scale=0.05, size=16))
    return np.asarray(rows)


def test_embedding_clustering_finds_the_families_and_still_names_them(corpus):
    result = clustering.cluster_postings(corpus, k=4, vectors=_family_vectors(corpus))
    assert result.representation == "embedding"
    assert result.k == 4
    # Every cluster is one family, so every cluster is pure.
    for cluster in range(4):
        titles = corpus.loc[result.labels == cluster, "title"]
        families = titles.str.replace(r"^(Junior|Senior) ", "", regex=True).unique()
        assert len(families) == 1
    # And the labels still come from words, not vector indices.
    assert all(isinstance(term, str) for term in result.summary["top_terms"].iloc[0])


def test_vectors_must_line_up_with_the_frame(corpus):
    with pytest.raises(ValueError, match="vectors for"):
        clustering.cluster_postings(corpus, k=3, vectors=np.zeros((5, 16)))


def test_comparison_reports_agreement_between_the_two_spaces(corpus):
    comparison = clustering.compare_representations(
        corpus, _family_vectors(corpus), k=4
    )
    assert -1.0 <= comparison.agreement <= 1.0
    table = comparison.as_table()
    assert "| tfidf |" in table and "| embedding |" in table
    assert "adjusted Rand index" in table
