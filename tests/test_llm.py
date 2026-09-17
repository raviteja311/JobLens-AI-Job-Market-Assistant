"""The LLM plumbing, tested without an LLM.

Every test here stubs the backend. A test that calls a real model is slow,
non-deterministic and offline-fragile, and none of what is being checked
here (JSON extraction, retry-on-validation-error, refusal detection) is about
the model's intelligence.
"""

import pytest
from pydantic import BaseModel, Field

from joblens.eval import generation
from joblens.llm import client, prompts
from joblens.rag import chat as chat_rag


class Answer(BaseModel):
    score: int = Field(ge=0, le=100)
    label: str = ""


class StubBackend:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, prompt, max_tokens, json_mode):
        self.prompts.append(prompt)
        return client.Completion(
            text=self.replies.pop(0),
            model="stub",
            backend="stub",
            prompt_tokens=10,
            completion_tokens=5,
        )


@pytest.fixture
def stub(monkeypatch):
    def install(replies):
        backend = StubBackend(replies)
        monkeypatch.setattr(client, "get_backend", lambda: backend)
        # The call log needs a database; these tests are about the parsing.
        monkeypatch.setattr(client, "_log_call", lambda **kwargs: None)
        return backend

    return install


def test_extract_json_from_a_fenced_block():
    assert client.extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_extract_json_from_prose():
    assert client.extract_json('Sure! {"a": 1} Hope that helps.') == '{"a": 1}'


def test_extract_json_leaves_bare_json_alone():
    assert client.extract_json('{"a": 1}') == '{"a": 1}'


def test_structured_output_parses_a_good_reply(stub):
    stub(['{"score": 80, "label": "good"}'])
    result = client.complete_structured("prompt", Answer, feature="test")
    assert result.value.score == 80
    assert result.attempts == 1
    assert result.repairs == []


def test_structured_output_retries_with_the_validation_error(stub):
    backend = stub(['{"score": "high"}', '{"score": 70, "label": "ok"}'])
    result = client.complete_structured("prompt", Answer, feature="test")
    assert result.value.score == 70
    assert result.attempts == 2
    # The retry has to carry the actual error, not just ask again.
    assert "could not be parsed" in backend.prompts[1]
    assert "score" in backend.prompts[1]


def test_structured_output_gives_up_with_a_useful_message(stub):
    stub(["nonsense", "still nonsense", "nope"])
    with pytest.raises(ValueError, match="no valid Answer after 3 attempts"):
        client.complete_structured("prompt", Answer, feature="test")


def test_out_of_range_values_are_rejected_not_clamped(stub):
    backend = stub(['{"score": 500}', '{"score": 50}'])
    result = client.complete_structured("prompt", Answer, feature="test")
    assert result.value.score == 50
    assert len(backend.prompts) == 2


def test_prompts_load_from_disk_with_a_version():
    prompt = prompts.load("chat_answer", "v1")
    assert prompt.version == "v1"
    assert "$sources" in prompt.template
    assert "$question" in prompt.template


def test_prompt_render_substitutes():
    prompt = prompts.load("chat_answer", "v1")
    rendered = prompt.render(sources="SRC", question="Q")
    assert "SRC" in rendered and "Q" in rendered
    assert "$sources" not in rendered


def test_prompt_render_survives_a_dollar_sign_in_the_data():
    # Postings are full of salaries. safe_substitute, not substitute.
    prompt = prompts.load("chat_answer", "v1")
    rendered = prompt.render(sources="pays $120k", question="how much?")
    assert "$120k" in rendered


def test_missing_prompt_version_lists_what_exists():
    with pytest.raises(FileNotFoundError, match="available: v1"):
        prompts.load("chat_answer", "v99")


def test_latest_picks_the_highest_version():
    versions = prompts.available("chat_answer")
    assert prompts.latest("chat_answer") == versions[-1]
    assert "v1" in versions


@pytest.mark.parametrize(
    "answer",
    [
        "I could not find postings that answer that.",
        "There is no information about the capital of Peru in the postings.",
        "The postings do not say who founded these companies.",
    ],
)
def test_refusals_are_detected(answer):
    assert generation.looks_like_refusal(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "St. Jude is hiring a Rust engineer [1].",
        "Four of the eight postings mention Kubernetes [2][3].",
    ],
)
def test_real_answers_are_not_mistaken_for_refusals(answer):
    assert not generation.looks_like_refusal(answer)


def test_no_answer_text_is_honest_about_the_corpus():
    assert "465" in chat_rag.NO_ANSWER
    assert "could not find" in chat_rag.NO_ANSWER.lower()
