import pandas as pd

from joblens.ml import skills


def test_finds_the_obvious_ones():
    found = skills.extract_skills(
        "Senior ML Engineer. PyTorch, Kubernetes and AWS. RAG experience a plus."
    )
    assert {"pytorch", "kubernetes", "aws", "rag"} <= set(found)


def test_aliases_map_to_one_canonical_name():
    assert skills.extract_skills("sklearn") == skills.extract_skills("scikit-learn")
    assert "hugging face" in skills.extract_skills("HuggingFace transformers")


def test_punctuated_names_survive_the_word_boundary():
    # The reason the pattern does not use \b: re.escape("c++") followed by \b
    # never matches, because '+' is already a non-word character.
    assert "c++" in skills.extract_skills("Strong C++ and Python")
    assert "c#" in skills.extract_skills("C# / .NET shop")
    assert "ci/cd" in skills.extract_skills("Owns CI/CD")


def test_does_not_match_inside_longer_words():
    # "go" inside "going", "r" inside anything, "java" inside "javascript".
    assert skills.extract_skills("We are going to grow the team") == []
    assert "java" not in skills.extract_skills("Frontend work in JavaScript")


def test_empty_text_is_not_an_error():
    assert skills.extract_skills(None) == []
    assert skills.extract_skills("") == []


def test_skill_matrix_shape_and_counts(corpus):
    matrix = skills.skill_matrix(corpus)
    assert len(matrix) == len(corpus)
    assert matrix.shape[1] == len(skills.SKILLS)
    assert matrix["python"].sum() > 0


def test_counts_are_sorted_and_non_zero(corpus):
    counts = skills.skill_counts(corpus)
    assert (counts > 0).all()
    assert counts.is_monotonic_decreasing
    assert counts.index[0] in {"python", "sql"}


def test_coverage_is_a_share(corpus):
    assert 0.0 <= skills.coverage(corpus) <= 1.0
    # Every synthetic posting names at least one known tool.
    assert skills.coverage(corpus) == 1.0


def test_coverage_of_a_posting_with_nothing_we_know():
    frame = pd.DataFrame(
        {"title": ["Barista"], "description": ["Make coffee, be friendly."]}
    )
    assert skills.coverage(frame) == 0.0


def test_tfidf_keywords_finds_the_distinguishing_term():
    keywords = skills.tfidf_keywords(
        [
            "kubernetes kubernetes deployment cluster",
            "tableau dashboard reporting",
        ],
        top_k=3,
    )
    # Bigrams are in the vocabulary too, so check the term appears rather than
    # demanding it be the unigram.
    assert any("kubernetes" in term for term in keywords[0])
    assert any("tableau" in term for term in keywords[1])


def test_encoder_columns_line_up_with_the_taxonomy():
    encoder = skills.SkillEncoder().fit(["python"])
    matrix = encoder.transform(pd.Series(["Python and Docker", "Nothing here"]))
    names = list(encoder.get_feature_names_out())
    assert matrix.shape == (2, len(skills.SKILLS))
    assert matrix[0][names.index("skill:docker")] == 1.0
    assert matrix[1].sum() == 0.0
