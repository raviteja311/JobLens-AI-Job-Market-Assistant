"""Gold v2: test labels built blind, per occurrence, with an additive pass.

Gold v1 was circular. It was built as `rules(posting) | adjudicated(teacher)`,
where adjudication filtered teacher terms through a hand-reviewed vocabulary
that is, in effect, the regex's own capability spec. Truth was therefore
defined as things the regex could have found, so the regex could not produce
a false positive and its precision came out at exactly 1.000 on all 60
postings. That is an identity, not a measurement.

Three changes fix it.

**Blind.** Candidates from both labellers are pooled and shuffled, and the
adjudicator never sees which system proposed a term. Provenance is stored in
a separate file and joined back afterwards, so it can be reported without
having influenced the judging.

**Per occurrence, not per term.** The decision is (posting, term) rather than
term. A global accept list is wrong in both directions: "go" is a language in
one posting and a verb in the next, "r" is a language and also half of "R&D".

**Additive.** A pass over the posting text adds skills neither labeller
found. Without it gold stays a subset of the union of the two labellers and
both recall figures are really recall-at-union, which flatters both.

The rules extractor's output is droppable like anything else. That single
property is what turns precision back into a measurement.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from joblens.finetune import dataset

DATA_DIR = dataset.DATA_DIR
GOLD_V2_FILE = DATA_DIR / "gold_v2.jsonl"
CANDIDATES_FILE = DATA_DIR / "gold_v2_candidates.json"
PROVENANCE_FILE = DATA_DIR / "gold_v2_provenance.json"
DECISIONS_FILE = DATA_DIR / "gold_v2_decisions.json"
SECOND_PASS_FILE = DATA_DIR / "gold_v2_second_pass.json"

KEEP = "keep"
DROP = "drop"
HUMAN_ONLY = "human_only"


@dataclass
class Candidate:
    """One term proposed for one posting, with no hint of where it came from."""

    posting_key: str
    term: str


@dataclass
class GoldPosting:
    posting_key: str
    posting_id: int
    title: str
    label: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)


def build_pool(examples: list[dataset.Example], seed: int = 20260919):
    """Pool rules and teacher candidates, shuffle, and split off provenance.

    Returns (candidates, provenance). The candidate list is what a human
    reads; the provenance map is written to a different file and is not
    consulted until after every decision is recorded.
    """
    rng = random.Random(seed)
    candidates: dict[str, list[str]] = {}
    provenance: dict[str, str] = {}
    for example in examples:
        rules = {s.lower() for s in example.rule_based}
        teacher = {s.lower() for s in example.teacher}
        pooled = sorted(rules | teacher)
        rng.shuffle(pooled)
        candidates[example.key] = pooled
        for term in pooled:
            if term in rules and term in teacher:
                origin = "both"
            elif term in rules:
                origin = "rules"
            else:
                origin = "teacher"
            provenance[f"{example.key}|{term}"] = origin
    return candidates, provenance


def save_pool(candidates: dict, provenance: dict) -> None:
    CANDIDATES_FILE.write_text(
        json.dumps(candidates, indent=1, sort_keys=True), encoding="utf-8"
    )
    PROVENANCE_FILE.write_text(
        json.dumps(provenance, indent=1, sort_keys=True), encoding="utf-8"
    )


def load_pool() -> tuple[dict, dict]:
    candidates = json.loads(CANDIDATES_FILE.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE_FILE.read_text(encoding="utf-8"))
    return candidates, provenance


def load_decisions(path: Path | None = None) -> dict:
    path = path or DECISIONS_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_decisions(decisions: dict, path: Path | None = None) -> Path:
    """Written after every posting so a half-finished pass is not lost."""
    path = path or DECISIONS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decisions, indent=1, sort_keys=True), encoding="utf-8")
    return path


def build_gold(examples: list[dataset.Example], decisions: dict) -> list[GoldPosting]:
    """Turn per-occurrence decisions into the final labels.

    `decisions` maps posting key to {"keep": [...], "drop": [...],
    "added": [...]}. Every candidate for a posting must appear in exactly one
    of keep or drop, so a posting cannot be silently half-judged.
    """
    candidates, _ = load_pool()
    gold: list[GoldPosting] = []
    for example in examples:
        verdict = decisions.get(example.key)
        if verdict is None:
            raise ValueError(f"no decision recorded for {example.key}")
        kept = {s.lower() for s in verdict.get("keep", [])}
        dropped = {s.lower() for s in verdict.get("drop", [])}
        added = {s.lower() for s in verdict.get("added", [])}

        offered = set(candidates.get(example.key, []))
        unjudged = offered - kept - dropped
        if unjudged:
            raise ValueError(
                f"{example.key}: candidates not judged: {sorted(unjudged)}"
            )
        stray = (kept | dropped) - offered
        if stray:
            raise ValueError(
                f"{example.key}: judged terms never offered: {sorted(stray)}"
            )

        gold.append(
            GoldPosting(
                posting_key=example.key,
                posting_id=example.posting_id,
                title=example.title,
                label=sorted(kept | added),
                dropped=sorted(dropped),
                added=sorted(added),
            )
        )
    return gold


def save_gold(gold: list[GoldPosting], path: Path | None = None) -> Path:
    path = path or GOLD_V2_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in gold:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
    return path


def load_gold(path: Path | None = None) -> list[GoldPosting]:
    path = path or GOLD_V2_FILE
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [GoldPosting(**json.loads(line)) for line in handle if line.strip()]


def content_hash(gold: list[GoldPosting]) -> str:
    """Freeze marker. Any change to any label changes this."""
    payload = json.dumps(
        [
            {"k": g.posting_key, "l": sorted(g.label)}
            for g in sorted(gold, key=lambda g: g.posting_key)
        ],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def apply_to_examples(
    examples: list[dataset.Example], gold: list[GoldPosting]
) -> list[dataset.Example]:
    """Overlay gold v2 labels onto Example objects for the scorer.

    Matched on (source, source_id) rather than primary key, because the
    postings table is a bigserial the test suite truncates.
    """
    by_key = {g.posting_key: g for g in gold}
    out = []
    for example in examples:
        row = by_key.get(example.key)
        if row is None:
            continue
        example.label = list(row.label)
        example.verified = True
        example.note = "gold v2: blind, per occurrence, with an additive pass"
        out.append(example)
    return out


def cohens_kappa(first: list[bool], second: list[bool]) -> float:
    """Agreement above chance on the binary keep/drop decision."""
    n = len(first)
    if n == 0:
        return 0.0
    observed = sum(1 for a, b in zip(first, second, strict=True) if a == b) / n
    pa = sum(first) / n
    pb = sum(second) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def agreement(first: dict, second: dict, keys: list[str]) -> dict:
    """Compare two independent passes over the same postings, per occurrence."""
    a_flags: list[bool] = []
    b_flags: list[bool] = []
    disagreements: list[str] = []
    for key in keys:
        one, two = first.get(key), second.get(key)
        if one is None or two is None:
            continue
        terms = sorted({s.lower() for s in one.get("keep", []) + one.get("drop", [])})
        for term in terms:
            kept_a = term in {s.lower() for s in one.get("keep", [])}
            kept_b = term in {s.lower() for s in two.get("keep", [])}
            a_flags.append(kept_a)
            b_flags.append(kept_b)
            if kept_a != kept_b:
                disagreements.append(f"{key}|{term}: {kept_a} vs {kept_b}")
    n = len(a_flags)
    exact = sum(1 for a, b in zip(a_flags, b_flags, strict=True) if a == b)
    return {
        "occurrences": n,
        "agreed": exact,
        "agreement": exact / n if n else 0.0,
        "kappa": cohens_kappa(a_flags, b_flags),
        "disagreements": disagreements,
    }


def provenance_report(gold: list[GoldPosting]) -> dict:
    """What the blind pass did, joined back to where each term came from.

    `rules_dropped` is the number that matters. If it is zero the regex still
    cannot produce a false positive and precision is an identity again.
    """
    _, provenance = load_pool()
    counts = {
        "kept_rules": 0,
        "kept_teacher": 0,
        "kept_both": 0,
        "dropped_rules": 0,
        "dropped_teacher": 0,
        "dropped_both": 0,
        "added_human_only": 0,
    }
    for row in gold:
        for term in row.label:
            origin = provenance.get(f"{row.posting_key}|{term}")
            if origin is None:
                counts["added_human_only"] += 1
            else:
                counts[f"kept_{origin}"] += 1
        for term in row.dropped:
            origin = provenance.get(f"{row.posting_key}|{term}", "teacher")
            counts[f"dropped_{origin}"] += 1
    return counts
