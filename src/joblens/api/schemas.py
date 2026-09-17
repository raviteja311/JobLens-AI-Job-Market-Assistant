"""Request and response shapes for the API.

Separate from the models the pipeline uses. `joblens.models.Posting` is what
we store; these are what we promise. Letting a storage model leak into a
response means the day a column is renamed, every client breaks.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    posting_id: int
    title: str
    company: str
    url: str
    location: str | None = None
    is_remote: bool = False
    score: float
    # Which retrievers found it and where. Exposed because a search UI that
    # cannot explain a result is a search UI nobody trusts.
    ranks: dict[str, int] = Field(default_factory=dict)
    snippet: str = ""


class SearchResponse(BaseModel):
    query: str
    mode: str
    reranked: bool
    took_ms: float
    results: list[SearchResult]


class Citation(BaseModel):
    """A source the answer is allowed to have used."""

    n: int
    posting_id: int
    title: str
    company: str
    url: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    limit: int = Field(default=8, ge=1, le=20)


class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    # True when retrieval found nothing worth answering from. The endpoint
    # says so rather than letting the model improvise, and the flag lets a
    # client render that case differently.
    grounded: bool
    took_ms: float
    cost_usd: float


class SkillGap(BaseModel):
    skill: str
    evidence: str = ""


class MatchResult(BaseModel):
    posting_id: int
    title: str
    company: str
    url: str
    fit_score: int = Field(ge=0, le=100)
    reasoning: str
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)


class MatchResponse(BaseModel):
    resume_skills: list[str]
    matches: list[MatchResult]
    took_ms: float
    cost_usd: float
