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
    # The Indian cities the location names, one spelling each
    # (places.cities_in), so a client can filter by city without parsing
    # "India - Bengaluru" and "Bangalore, Karnataka" itself.
    cities: list[str] = Field(default_factory=list)
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
    """One posting the model was given, numbered as the answer cites it."""

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
    # The postings the answer cites with [n], and nothing else. Listing every
    # retrieved posting here claimed support the answer never used.
    citations: list[Citation]
    # Everything the model was given, cited or not, so a client can still
    # show what retrieval found.
    retrieved: list[Citation] = Field(default_factory=list)
    # True when the answer cites at least one supplied posting and invents
    # no citation number. False when retrieval found nothing worth answering
    # from (the model is not called and the reply is a refusal), when the
    # model answered without citing anything, or when it cited a number it
    # was never given; the flag lets a client render those cases differently.
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
