"""Building the skill-extraction training set by distillation.

The pattern is the standard one: a big model labels, a small model learns.
The plan calls for a frontier API model as the teacher; there is no API key
configured, so the teacher here is llama3.1 8B over Ollama. That is a
meaningfully weaker teacher and every number downstream inherits the
difference, which is said plainly in the README rather than buried.

Two decisions that keep the evaluation honest.

**The teacher's labels train the student, they do not score it.** Scoring a
distilled student against its own teacher's output measures imitation, and
the teacher wins 1.0 by construction. The test split is hand-corrected first
and that is what every model is scored on.

**Hand-correction adjudicates disagreements.** Where the Phase 2 dictionary
and the teacher agree, the label stands. Where they disagree, a human reads
the posting and decides. Cheap and high-precision, with one blind spot worth
stating: a skill that both of them miss is never surfaced for review, so
recall on the test set is recall against what one of the two systems saw.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from joblens.cleaning import strip_noise
from joblens.llm import client, prompts
from joblens.ml.skills import SKILLS, extract_skills

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "finetune"
TRAIN_FILE = DATA_DIR / "train.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"

# How much of a posting the models see. Long enough to carry a requirements
# list, short enough that a 0.5B student with a 1k window has room to answer.
MAX_DESCRIPTION_CHARS = 1800

KNOWN = set(SKILLS)


class ExtractedSkills(BaseModel):
    """The only shape the teacher is allowed to return."""

    skills: list[str] = Field(default_factory=list)

    @field_validator("skills")
    @classmethod
    def _tidy(cls, values: list[str]) -> list[str]:
        seen: list[str] = []
        for value in values:
            skill = str(value).strip().lower()
            if skill and skill not in seen:
                seen.append(skill)
        return seen


@dataclass
class Example:
    posting_id: int
    source: str
    source_id: str
    title: str
    description: str
    # What the teacher said, what the Phase 2 dictionary said, and the label
    # actually used. Keeping all three is what makes the disagreement
    # adjudication reproducible instead of a claim.
    teacher: list[str] = field(default_factory=list)
    rule_based: list[str] = field(default_factory=list)
    label: list[str] = field(default_factory=list)
    verified: bool = False
    note: str = ""

    @property
    def key(self) -> str:
        return f"{self.source}:{self.source_id}"

    @property
    def disagreement(self) -> set[str]:
        return set(self.teacher) ^ set(self.rule_based)


def load_postings(conn, limit: int | None = None) -> list[dict]:
    rows = conn.execute("""
        select id, source, source_id, title, description
          from postings
         where length(coalesce(description, '')) > 200
         order by id
        """).fetchall()
    return rows[:limit] if limit else rows


def render(title: str, description: str) -> str:
    """The exact text every model sees. One function so teacher, student and
    baseline cannot drift apart without the diff showing it."""
    body = strip_noise(description or "")[:MAX_DESCRIPTION_CHARS]
    return f"Title: {title}\n\n{body}"


def label_with_teacher(title: str, description: str) -> tuple[list[str], float]:
    prompt = prompts.load("extract_skills")
    result = client.complete_structured(
        prompt.render(
            title=title,
            description=strip_noise(description or "")[:MAX_DESCRIPTION_CHARS],
        ),
        ExtractedSkills,
        feature="distil_label",
        prompt=prompt,
        max_tokens=300,
    )
    return result.value.skills, result.completion.cost_usd


LABELLED_FILE = DATA_DIR / "labelled.jsonl"


def label_corpus(conn, limit: int = 360, path: Path | None = None) -> list[Example]:
    """Label postings with the teacher, appending as it goes.

    Written incrementally and resumable because this takes over an hour on a
    local 8B model. Losing seventy minutes of teacher calls to one bad row is
    the kind of thing that only has to happen once.
    """
    path = path or LABELLED_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    # Resume on (source, source_id), never on postings.id. The primary key is
    # a bigserial and the test suite truncates the table, so a re-ingest
    # renumbers every row. Resuming on the id would then skip postings that
    # happen to have inherited a used number and re-label ones that did not,
    # silently, after an hour of teacher calls. The golden set learned this in
    # Phase 3; this module had the same bug until the corpus was rebuilt under
    # it.
    done = {e.key for e in load(path)}
    rows = [
        r
        for r in load_postings(conn, limit)
        if f"{r['source']}:{r['source_id']}" not in done
    ]
    if done:
        log.info("resuming: %s already labelled, %s to go", len(done), len(rows))

    with path.open("a", encoding="utf-8") as handle:
        for i, row in enumerate(rows, start=1):
            try:
                teacher, _ = label_with_teacher(row["title"], row["description"])
            except Exception as exc:  # noqa: BLE001 - one bad row must not stop it
                log.warning("teacher failed on posting %s: %s", row["id"], exc)
                continue
            example = Example(
                posting_id=row["id"],
                source=row["source"],
                source_id=row["source_id"],
                title=row["title"],
                description=row["description"] or "",
                teacher=teacher,
                rule_based=extract_skills(render(row["title"], row["description"])),
                label=teacher,
            )
            handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
            handle.flush()
            if i % 25 == 0:
                log.info("labelled %s of %s", i, len(rows))
    return load(path)


def split(
    examples: list[Example], test_size: int = 60, seed: int = 42
) -> tuple[list[Example], list[Example]]:
    """Shuffle and split. Done before any hand-correction, so nothing about
    the test set can leak into training through a label fixed twice."""
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    return shuffled[test_size:], shuffled[:test_size]


def split_hash(examples: list[Example]) -> str:
    """A fingerprint of which postings are in a split, in order.

    Recorded next to every training run. Two runs quoting the same hash were
    trained on the same rows; two quoting different hashes are not
    comparable, however similar their configs look.
    """
    joined = "|".join(e.key for e in examples)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def train_excluding(labelled: list[Example], test: list[Example]) -> list[Example]:
    """Everything labelled that is not in the frozen test set.

    Used when the corpus grows. The test split was drawn once, corrected by
    hand once, and every number in the comparison table is measured on it, so
    it must not move when more training data arrives. Exclusion is by
    (source, source_id) rather than by primary key, because a re-ingest
    renumbers the table.
    """
    held = {e.key for e in test}
    return [e for e in labelled if e.key not in held]


def balance(
    examples: list[Example], max_empty_share: float = 0.3, seed: int = 42
) -> list[Example]:
    """Cap the share of examples whose correct answer is an empty list.

    The first run collapsed: 56% of the training labels were empty, and a
    0.5B model given 163 examples found the obvious local optimum of always
    answering `{"skills": []}`. It scored micro F1 0.000 and macro F1 0.617,
    the whole macro score coming from postings where saying nothing happens
    to be right.

    Empty examples are not noise and cannot all be dropped. A third of this
    corpus is genuinely non-technical, and a model that never answers empty
    would invent skills for marketing roles. The cap keeps enough to teach
    "sometimes the answer is nothing" without making it the safest guess.
    """
    empty = [e for e in examples if not e.label]
    filled = [e for e in examples if e.label]
    if not filled:
        return list(examples)

    keep = int(len(filled) * max_empty_share / (1 - max_empty_share))
    rng = random.Random(seed)
    rng.shuffle(empty)
    balanced = filled + empty[:keep]
    rng.shuffle(balanced)
    return balanced


def save(examples: list[Example], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
    return path


def load(path: Path) -> list[Example]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [Example(**json.loads(line)) for line in handle if line.strip()]


def to_chat(example: Example) -> dict:
    """One training row in the chat format trl expects.

    The assistant turn is the JSON the student has to learn to produce. It is
    sorted so the model is not asked to also memorise an arbitrary ordering,
    which is signal it cannot learn and loss it cannot reduce.
    """
    prompt = prompts.load("extract_skills")
    user = prompt.render(
        title=example.title,
        description=strip_noise(example.description)[:MAX_DESCRIPTION_CHARS],
    )
    answer = json.dumps({"skills": sorted(example.label)}, ensure_ascii=False)
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ]
    }


def stats(examples: list[Example]) -> dict:
    if not examples:
        return {"examples": 0}
    teacher_only = sum(len(set(e.teacher) - set(e.rule_based)) for e in examples)
    rule_only = sum(len(set(e.rule_based) - set(e.teacher)) for e in examples)
    off_taxonomy = {s for e in examples for s in e.teacher if s not in KNOWN}
    return {
        "examples": len(examples),
        "verified": sum(1 for e in examples if e.verified),
        "mean_labels": round(sum(len(e.label) for e in examples) / len(examples), 2),
        "empty_labels": sum(1 for e in examples if not e.label),
        "teacher_only_mentions": teacher_only,
        "rule_only_mentions": rule_only,
        # Skills the teacher invented that the Phase 2 taxonomy has never
        # heard of. A large number here is the teacher hallucinating, or the
        # taxonomy being out of date, and the two look identical from a count.
        "off_taxonomy_skills": len(off_taxonomy),
        "off_taxonomy_examples": sorted(off_taxonomy)[:15],
    }
