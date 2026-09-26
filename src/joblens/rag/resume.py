"""Resume in, ranked jobs out, with the gaps named.

Three steps and each one can fail differently. PDF extraction fails on
scanned resumes and says so rather than passing an empty string down the
chain. Skill extraction is an LLM call validated against a pydantic model.
Matching is one LLM call per posting, which is the expensive part, so
retrieval narrows the field first and the count is capped.

The output people actually want is `missing_skills`. Fit scores are easy to
produce and hard to trust; "this posting wants Kubernetes and your resume
does not mention it" is checkable by the person reading it.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, field_validator

from joblens.llm import client, prompts
from joblens.ml.skills import extract_skills
from joblens.search import retrieval
from joblens.search.embeddings import Embedder

log = logging.getLogger(__name__)

MAX_RESUME_CHARS = 12_000
MAX_MATCHES = 8


class ResumeProfile(BaseModel):
    """What the model is allowed to return for a resume."""

    skills: list[str] = Field(default_factory=list)
    seniority: str = "mid"
    years_experience: float = 0
    summary: str = ""

    @field_validator("skills")
    @classmethod
    def _tidy(cls, values: list[str]) -> list[str]:
        seen: list[str] = []
        for value in values:
            skill = str(value).strip().lower()
            if skill and skill not in seen:
                seen.append(skill)
        return seen

    @field_validator("seniority")
    @classmethod
    def _bucket(cls, value: str) -> str:
        value = str(value).strip().lower()
        return value if value in {"junior", "mid", "senior"} else "mid"


class JobMatch(BaseModel):
    fit_score: int = Field(ge=0, le=100)
    reasoning: str = ""
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)


@dataclass
class Match:
    posting_id: int
    title: str
    company: str
    url: str
    verdict: JobMatch


@dataclass
class MatchReport:
    profile: ResumeProfile
    matches: list[Match] = field(default_factory=list)
    took_ms: int = 0
    cost_usd: float = 0.0
    rule_based_skills: list[str] = field(default_factory=list)


def pdf_to_text(data: bytes) -> str:
    """Extract text from a PDF, or explain why there is none."""
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ValueError(
                "that PDF is password protected. Remove the password and try again."
            )
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except PyPdfError as exc:
        # A truncated download, a .docx renamed to .pdf, or a file that is not
        # a PDF at all. The caller's mistake, so it must not be a 500.
        raise ValueError(f"could not read that PDF: {exc}") from exc
    text = text.strip()
    if not text:
        raise ValueError(
            "no text in that PDF. It is probably a scan, and this project does "
            "not do OCR. Export the resume as text-based PDF and try again."
        )
    return text[:MAX_RESUME_CHARS]


def extract_profile(resume_text: str, prompt_version: str | None = None):
    """LLM skill extraction, validated. The Phase 2 table is the baseline."""
    prompt = prompts.load("resume_skills", prompt_version)
    result = client.complete_structured(
        prompt.render(resume=resume_text),
        ResumeProfile,
        feature="resume_skills",
        prompt=prompt,
        max_tokens=800,
    )
    return result


def match_resume(
    conn,
    resume_text: str,
    embedder: Embedder | None = None,
    limit: int = MAX_MATCHES,
    prompt_version: str | None = None,
) -> MatchReport:
    began = time.perf_counter()
    limit = min(limit, MAX_MATCHES)

    extracted = extract_profile(resume_text, prompt_version)
    profile = extracted.value
    cost = extracted.completion.cost_usd

    # The Phase 2 dictionary runs on the same text. It costs nothing and it is
    # the control: when the two disagree badly, one of them is wrong and the
    # logs say which call produced the disagreement.
    rule_based = extract_skills(resume_text)

    query = " ".join(profile.skills[:15]) or resume_text[:400]
    hits = retrieval.search(
        conn, query, mode="hybrid", embedder=embedder, strategy="whole", limit=limit
    )
    if not hits:
        return MatchReport(
            profile=profile,
            took_ms=int((time.perf_counter() - began) * 1000),
            cost_usd=cost,
            rule_based_skills=rule_based,
        )

    rows = conn.execute(
        "select id, description, location from postings where id = any(%s)",
        ([h.posting_id for h in hits],),
    ).fetchall()
    details = {r["id"]: r for r in rows}

    prompt = prompts.load("job_match", prompt_version)
    matches: list[Match] = []
    for hit in hits:
        row = details.get(hit.posting_id, {})
        rendered = prompt.render(
            skills=", ".join(profile.skills) or "none listed",
            summary=profile.summary or "not stated",
            title=hit.title,
            company=hit.company,
            location=row.get("location") or "not stated",
            description=(row.get("description") or "")[:2000],
        )
        try:
            result = client.complete_structured(
                rendered, JobMatch, feature="match", prompt=prompt, max_tokens=500
            )
        except ValueError:
            # One posting the model cannot score must not lose the other seven.
            log.warning("could not score posting %s, skipping", hit.posting_id)
            continue
        cost += result.completion.cost_usd
        matches.append(
            Match(
                posting_id=hit.posting_id,
                title=hit.title,
                company=hit.company,
                url=hit.url,
                verdict=result.value,
            )
        )

    matches.sort(key=lambda m: m.verdict.fit_score, reverse=True)
    return MatchReport(
        profile=profile,
        matches=matches,
        took_ms=int((time.perf_counter() - began) * 1000),
        cost_usd=cost,
        rule_based_skills=rule_based,
    )
