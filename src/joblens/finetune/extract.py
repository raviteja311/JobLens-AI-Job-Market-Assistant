"""Four skill extractors behind one interface.

    rules       the Phase 2 dictionary. Free, instant, knows 80 skills.
    base        Qwen2.5-0.5B-Instruct, no fine-tuning. The control.
    tuned       the same model with the LoRA adapter.
    teacher     llama3.1 8B over Ollama, the model that produced the labels.

They all implement `extract(title, description) -> list[str]` and all record
latency, so the comparison table is one loop rather than four scripts. Which
one `/match` uses is a config flag, which is the point: the fine-tuned model
is only interesting if it can be dropped into the place the big one occupied.

The student is served through transformers rather than Ollama. Getting a LoRA
adapter into Ollama means merging and converting to GGUF, which is a
toolchain to install and a second copy of the weights to keep in sync for a
model this small. The interface is what matters here, not the runtime.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from joblens.cleaning import strip_noise
from joblens.finetune.dataset import MAX_DESCRIPTION_CHARS, ExtractedSkills, render
from joblens.llm import client, prompts
from joblens.ml.skills import extract_skills as rule_based_extract

log = logging.getLogger(__name__)

ADAPTER_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "skill-lora"


@dataclass
class Usage:
    calls: int = 0
    seconds: float = 0.0
    usd: float = 0.0
    failures: int = 0

    @property
    def ms_per_call(self) -> float:
        return self.seconds * 1000 / self.calls if self.calls else 0.0


class Extractor:
    name: str
    usage: Usage

    def extract(self, title: str, description: str) -> list[str]:
        raise NotImplementedError


def _normalise(skills: list[str]) -> list[str]:
    """Lowercase, dedupe, drop blanks. Applied to every extractor's output so
    the comparison is about what they found, not how they capitalised it."""
    seen: list[str] = []
    for skill in skills:
        value = str(skill).strip().lower()
        if value and value not in seen:
            seen.append(value)
    return sorted(seen)


class RuleExtractor(Extractor):
    """The Phase 2 dictionary. The number everything else has to beat."""

    name = "rules"

    def __init__(self) -> None:
        self.usage = Usage()

    def extract(self, title: str, description: str) -> list[str]:
        began = time.perf_counter()
        # dataset.render, not skills.posting_text. render truncates the
        # description to MAX_DESCRIPTION_CHARS and posting_text does not, so
        # the two see different amounts of a long posting. The labels were
        # built from render's output, which meant the baseline was scored on
        # more text than the gold was built from and charged with false
        # positives for skills it had correctly found in the tail.
        #
        # render's docstring already said it was "the exact text every model
        # sees ... so teacher, student and baseline cannot drift apart". The
        # baseline was the one not calling it.
        found = rule_based_extract(render(title, description))
        self.usage.calls += 1
        self.usage.seconds += time.perf_counter() - began
        return _normalise(found)


class TeacherExtractor(Extractor):
    """llama3.1 8B over Ollama, through the Phase 4 client so it is logged."""

    name = "teacher"

    def __init__(self, feature: str = "distil_eval") -> None:
        self.usage = Usage()
        self.feature = feature

    def extract(self, title: str, description: str) -> list[str]:
        prompt = prompts.load("extract_skills")
        began = time.perf_counter()
        try:
            result = client.complete_structured(
                prompt.render(
                    title=title,
                    description=strip_noise(description or "")[:MAX_DESCRIPTION_CHARS],
                ),
                ExtractedSkills,
                feature=self.feature,
                prompt=prompt,
                max_tokens=300,
            )
        except Exception:  # noqa: BLE001 - a refusal to parse is a real result
            self.usage.calls += 1
            self.usage.failures += 1
            self.usage.seconds += time.perf_counter() - began
            return []
        self.usage.calls += 1
        self.usage.seconds += time.perf_counter() - began
        self.usage.usd += result.completion.cost_usd
        return _normalise(result.value.skills)


_JSON = re.compile(r"\{.*\}", re.S)


@dataclass
class LocalExtractor(Extractor):
    """Qwen2.5-0.5B through transformers, with or without the LoRA adapter."""

    adapter: Path | None = None
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_new_tokens: int = 160
    usage: Usage = field(default_factory=Usage)
    _model: object | None = field(default=None, repr=False)
    _tokenizer: object | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return "tuned" if self.adapter else "base"

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            log.info("loading %s (adapter=%s)", self.base_model, self.adapter)
            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model, dtype=torch.float32
            )
            if self.adapter:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, str(self.adapter))
                model = model.merge_and_unload()
            model.eval()
            self._model = model
        return self._model, self._tokenizer

    def extract(self, title: str, description: str) -> list[str]:
        import torch

        model, tokenizer = self._load()
        prompt = prompts.load("extract_skills")
        user = prompt.render(
            title=title,
            description=strip_noise(description or "")[:MAX_DESCRIPTION_CHARS],
        )
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)

        began = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        self.usage.calls += 1
        self.usage.seconds += time.perf_counter() - began

        match = _JSON.search(reply)
        if not match:
            # The base model does this constantly and the tuned one should
            # not. Counted rather than hidden: "returns unparseable output"
            # is the single most useful difference between the two.
            self.usage.failures += 1
            return []
        try:
            payload = json.loads(match.group(0))
            return _normalise(ExtractedSkills(**payload).skills)
        except Exception:  # noqa: BLE001
            self.usage.failures += 1
            return []


def get_extractor(name: str, adapter: Path | None = None) -> Extractor:
    if name == "rules":
        return RuleExtractor()
    if name == "teacher":
        return TeacherExtractor()
    if name == "base":
        return LocalExtractor(adapter=None)
    if name == "tuned":
        path = adapter or ADAPTER_DIR
        if not path.exists():
            raise FileNotFoundError(
                f"no adapter at {path}. Run `python -m joblens distil-train` first."
            )
        return LocalExtractor(adapter=path)
    raise ValueError(f"unknown extractor {name!r}. known: rules, base, tuned, teacher")
