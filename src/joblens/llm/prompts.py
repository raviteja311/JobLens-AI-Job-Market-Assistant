"""Prompts live in files, with versions, and are never inlined.

The rule is not tidiness. A prompt is the most-changed and least-reviewed
part of an LLM feature: inline it as an f-string and a one-word edit ships
with no diff anyone reads, no way to say which version produced Tuesday's
answer, and no way to A/B two variants. On disk with a version in the
filename, `git log prompts/` is the change history and the A/B harness in
Phase 5 is a loop over two directory entries.

`string.Template` rather than str.format or f-strings: these prompts contain
JSON examples full of braces, and every brace would have to be doubled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Template

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    template: str

    def render(self, **values: object) -> str:
        # safe_substitute, not substitute: a posting that happens to contain
        # a dollar sign must not raise KeyError halfway through a request.
        return Template(self.template).safe_substitute(**values)


def available(name: str) -> list[str]:
    directory = PROMPTS_DIR / name
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.md"))


def latest(name: str) -> str:
    versions = available(name)
    if not versions:
        raise FileNotFoundError(f"no prompt versions found in {PROMPTS_DIR / name}")
    return versions[-1]


def load(name: str, version: str | None = None) -> Prompt:
    version = version or latest(name)
    path = PROMPTS_DIR / name / f"{version}.md"
    if not path.exists():
        known = ", ".join(available(name)) or "none"
        raise FileNotFoundError(f"no prompt {name}/{version}. available: {known}")
    return Prompt(name=name, version=version, template=path.read_text(encoding="utf-8"))
