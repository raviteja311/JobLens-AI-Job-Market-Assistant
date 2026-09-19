from joblens.search import chunking

LONG = "\n\n".join(f"Paragraph {i} " + "filler words here. " * 20 for i in range(6))


def test_whole_is_one_chunk():
    chunks = chunking.chunk_whole(1, "ML Engineer", "Acme", "We use PyTorch.")
    assert len(chunks) == 1
    assert chunks[0].strategy == "whole"
    assert "ML Engineer at Acme" in chunks[0].content
    assert "PyTorch" in chunks[0].content


def test_header_does_not_eat_letters_off_the_title():
    # str.strip(" at") strips any of ' ', 'a', 't', which turned
    # "Data Analyst at " into "Data Analys".
    chunks = chunking.chunk_whole(1, "Data Analyst", "", "")
    assert chunks[0].content.startswith("Data Analyst")


def test_sections_split_a_long_posting():
    chunks = chunking.chunk_sections(1, "ML Engineer", "Acme", LONG)
    assert len(chunks) > 1
    assert all(c.strategy == "section" for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_every_section_carries_the_header():
    # A chunk saying "5 years of Python" is useless without knowing the job.
    for chunk in chunking.chunk_sections(1, "ML Engineer", "Acme", LONG):
        assert chunk.content.startswith("ML Engineer at Acme")


def test_sections_stay_under_the_window():
    for chunk in chunking.chunk_sections(1, "ML Engineer", "Acme", LONG):
        body = chunk.content.split(". ", 1)[-1]
        assert len(body) <= chunking.MAX_CHARS + 100


def test_a_single_huge_paragraph_is_hard_split():
    chunks = chunking.chunk_sections(1, "Role", "Co", "word " * 1000)
    assert len(chunks) > 1


def test_two_chunks_with_a_tiny_tail_merge_without_indexerror():
    # The tail-merge used to be written as
    # chunks[-2] = f"{chunks[-2]} {chunks.pop()}", which evaluates the pop
    # first and then indexes [-2] on the shortened list.
    body = ("filler " * 200) + "\n\n" + "Apply soon."
    chunks = chunking.chunk_sections(1, "Role", "Co", body)
    assert chunks
    assert "Apply soon." in chunks[-1].content


def test_empty_description_still_produces_one_chunk():
    chunks = chunking.chunk_sections(1, "Role", "Co", "")
    assert len(chunks) == 1
    assert chunks[0].content == "Role at Co"


def test_boilerplate_is_stripped_before_embedding():
    body = (
        "We need a Rust engineer. Apply at https://jobs.example.com/1 . "
        "Please mention the word FUTURESTIC when applying to show you read "
        "the job post completely."
    )
    content = chunking.chunk_whole(1, "Engineer", "Co", body)[0].content
    assert "FUTURESTIC" not in content
    assert "jobs.example.com" not in content
    assert "Rust engineer" in content


def test_unknown_strategy_names_the_known_ones():
    try:
        chunking.chunk("sentences", 1, "t", "c", "d")
    except ValueError as exc:
        assert "whole" in str(exc) and "section" in str(exc)
    else:
        raise AssertionError("expected ValueError")
