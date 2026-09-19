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
