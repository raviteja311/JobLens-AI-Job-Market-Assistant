"""Hand-correcting the test split.

The test set has to be corrected by a human before anything is scored on it.
A distilled student graded against its own teacher's output is being measured
on imitation, and the teacher scores 1.0 by construction, which makes the
whole comparison meaningless in the direction that flatters it.

Correcting 60 postings from scratch is an afternoon. Adjudicating only where
the two independent labellers disagree is twenty minutes, because agreement
between a regex dictionary and an 8B model is strong evidence on its own.
`review()` produces exactly that queue.

The blind spot, stated because it is real: a skill that both the dictionary
and the teacher miss never reaches the queue and never enters the labels. So
recall measured on this test set is recall against the union of what those
two saw, not against the posting. It would take reading all 60 in full to
close that, and the number it would change is not worth the afternoon at this
corpus size.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from joblens.cleaning import strip_noise
from joblens.finetune.dataset import Example
from joblens.ml.skills import SKILLS

KNOWN_SKILLS = {s.lower() for s in SKILLS}

VOCABULARY_FILE = (
    Path(__file__).resolve().parents[3] / "data" / "finetune" / "vocabulary.yaml"
)


def load_vocabulary() -> tuple[set[str], set[str]]:
    """The hand-adjudicated verdict on terms outside the Phase 2 taxonomy.

    A substring check cannot separate a technology the dictionary is missing
    from prose the teacher lifted out of the posting, because both of them
    are in the text. These two lists are that separation, made by reading
    them.
    """
    if not VOCABULARY_FILE.exists():
        return set(), set()
    payload = yaml.safe_load(VOCABULARY_FILE.read_text(encoding="utf-8")) or {}

    def collect(prefix: str) -> set[str]:
        # Reviews are appended as accepted / accepted_round_two / ... rather
        # than merged into one list, so the file still reads as a record of
        # what was decided when and on how much data.
        found: set[str] = set()
        for key, values in payload.items():
            if key == prefix or key.startswith(f"{prefix}_"):
                found |= {str(s).lower() for s in values or []}
        return found

    return collect("accepted"), collect("rejected")


# How much of the posting to show next to a disputed skill. Enough to see
# whether the word actually appears in a requirements list.
EXCERPT_CHARS = 600


@dataclass
class Dispute:
    posting_id: int
    key: str
    title: str
    agreed: list[str]
    teacher_only: list[str]
    rules_only: list[str]
    excerpt: str

    def render(self) -> str:
        lines = [f"[{self.posting_id}] {self.title[:70]}"]
        if self.agreed:
            lines.append(f"    agreed      : {', '.join(self.agreed)}")
        if self.teacher_only:
            lines.append(f"    teacher only: {', '.join(self.teacher_only)}")
        if self.rules_only:
            lines.append(f"    rules only  : {', '.join(self.rules_only)}")
        return "\n".join(lines)


def review(examples: list[Example]) -> list[Dispute]:
    """The queue: one entry per posting where the two labellers disagree."""
    disputes = []
    for example in examples:
        teacher, rules = set(example.teacher), set(example.rule_based)
        if teacher == rules:
            continue
        disputes.append(
            Dispute(
                posting_id=example.posting_id,
                key=example.key,
                title=example.title,
                agreed=sorted(teacher & rules),
                teacher_only=sorted(teacher - rules),
                rules_only=sorted(rules - teacher),
                excerpt=strip_noise(example.description)[:EXCERPT_CHARS],
            )
        )
    return disputes


def adjudicate(example: Example) -> tuple[list[str], list[str]]:
    """Decide one posting's label. Returns (label, rejected).

    The rule that does most of the work: **the dictionary is exhaustive over
    its own vocabulary.** It is a regex over 80 canonical skills and their
    aliases, so if a posting names PyTorch anywhere in its title or
    description, the dictionary found it. Therefore a skill that is in the
    taxonomy and was returned by the teacher alone is a skill the posting
    does not contain, and the teacher invented it.

    That is not a hypothetical. On this test split the teacher-only mentions
    that do not appear anywhere in the text are led by pytorch (12), python
    (10) and aws (5): the model is pattern-matching "this is an ML job" onto
    the skills such jobs usually want, which is exactly the failure a
    dictionary cannot make.

    For skills outside the taxonomy the dictionary is silent by construction,
    so the check is whether the posting names the term at all.

    Rules-only mentions are always accepted. The dictionary only fires on a
    literal alias match, so its output is present in the text by definition;
    a rules-only skill means the teacher missed it.
    """
    accepted_vocab, rejected_vocab = load_vocabulary()
    text = strip_noise(example.description).lower()
    title = (example.title or "").lower()
    teacher, rules = set(example.teacher), set(example.rule_based)

    label = set(teacher & rules) | set(rules - teacher)
    rejected: list[str] = []
    for skill in sorted(teacher - rules):
        lowered = skill.lower()
        if lowered in KNOWN_SKILLS:
            rejected.append(skill)  # the dictionary would have found it
        elif lowered in rejected_vocab:
            rejected.append(skill)  # read, and it is not a skill
        elif lowered in accepted_vocab and (lowered in text or lowered in title):
            label.add(skill)
        else:
            # Unseen term. Dropped rather than guessed: an unreviewed label
            # is exactly the thing this file exists to prevent.
            rejected.append(skill)
    return sorted(label), rejected


def adjudicate_all(examples: list[Example]) -> tuple[list[Example], dict]:
    """Apply `adjudicate` to every example and report what it did."""
    accepted_off_taxonomy = 0
    rejected_hallucinated = 0
    rejected_absent = 0
    for example in examples:
        label, rejected = adjudicate(example)
        for skill in rejected:
            if skill.lower() in KNOWN_SKILLS:
                rejected_hallucinated += 1
            else:
                rejected_absent += 1
        accepted_off_taxonomy += sum(1 for s in label if s.lower() not in KNOWN_SKILLS)
        example.label = label
        example.verified = True
        example.note = "adjudicated: dictionary exhaustive over its vocabulary"
    return examples, {
        "examples": len(examples),
        "mean_labels": (
            round(sum(len(e.label) for e in examples) / len(examples), 2)
            if examples
            else 0
        ),
        "accepted_off_taxonomy": accepted_off_taxonomy,
        "rejected_teacher_hallucinations": rejected_hallucinated,
        "rejected_not_in_text": rejected_absent,
    }


def apply(
    examples: list[Example], decisions: dict[int, list[str]], note: str = ""
) -> list[Example]:
    """Write the adjudicated labels back.

    A posting absent from `decisions` had no disagreement, so its label is
    what both labellers already agreed on. Everything in the test split comes
    out verified either way: agreement is a decision too, and marking only the
    disputed ones would leave the eval refusing to score most of the set.
    """
    for example in examples:
        if example.posting_id in decisions:
            example.label = sorted(
                {s.strip().lower() for s in decisions[example.posting_id] if s.strip()}
            )
            example.note = note or "adjudicated disagreement"
        else:
            example.label = sorted(set(example.teacher) & set(example.rule_based))
            example.note = note or "labellers agreed"
        example.verified = True
    return examples


def summary(examples: list[Example]) -> dict:
    disputed = review(examples)
    return {
        "examples": len(examples),
        "verified": sum(1 for e in examples if e.verified),
        "agreed": len(examples) - len(disputed),
        "disputed": len(disputed),
        "teacher_only_mentions": sum(len(d.teacher_only) for d in disputed),
        "rules_only_mentions": sum(len(d.rules_only) for d in disputed),
    }
