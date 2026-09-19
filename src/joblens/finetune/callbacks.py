"""Training-time instrumentation, written because loss lied.

The first run's eval loss fell every single epoch, 1.600 to 1.573 to 1.565,
and produced a model with a micro F1 of exactly zero. Nothing in the training
output hinted at it. Loss going down told us the model was getting better at
predicting the tokens it was being scored on, and since those tokens were
overwhelmingly `{"skills": []}`, it was.

So the metric that decides which checkpoint to keep is F1 on generated
output, not loss, and the share of degenerate answers is logged as a
first-class number next to it. A run that collapses now says so on the
epoch it happens.

Generation during training is slow on CPU, so the validation slice is small
and held out of training rather than borrowed from the test set. It is there
to catch a collapse, not to measure quality; the frozen 60-posting test set
does that once, at the end.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from transformers import TrainerCallback

from joblens.cleaning import strip_noise
from joblens.finetune.dataset import MAX_DESCRIPTION_CHARS, Example
from joblens.llm import prompts

log = logging.getLogger(__name__)


@dataclass
class EpochScore:
    epoch: float
    micro_f1: float
    empty_share: float
    unparseable: int
    eval_loss: float | None = None

    def summary(self) -> str:
        loss = f"{self.eval_loss:.4f}" if self.eval_loss is not None else "n/a"
        return (
            f"epoch {self.epoch:.0f}: micro F1 {self.micro_f1:.3f}, "
            f"empty {self.empty_share:.0%}, unparseable {self.unparseable}, "
            f"eval loss {loss}"
        )


def _micro_f1(predictions: list[set[str]], truths: list[set[str]]) -> float:
    tp = sum(len(p & t) for p, t in zip(predictions, truths, strict=True))
    npred = sum(len(p) for p in predictions)
    ntruth = sum(len(t) for t in truths)
    if not npred or not ntruth:
        return 0.0
    precision, recall = tp / npred, tp / ntruth
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


@dataclass
class SkillF1Callback(TrainerCallback):
    """Generate on a validation slice each epoch and score what comes out.

    Keeps the adapter from the best-F1 epoch. `load_best_model_at_end` would
    do this for loss, which is precisely the number that cannot be trusted
    here.
    """

    examples: list[Example]
    tokenizer: object
    max_new_tokens: int = 64
    history: list[EpochScore] = field(default_factory=list)
    best_f1: float = -1.0
    best_epoch: float = 0.0
    _best_state: dict | None = field(default=None, repr=False)

    def _generate(self, model, example: Example) -> str:
        import torch

        prompt = prompts.load("extract_skills")
        user = prompt.render(
            title=example.title,
            description=strip_noise(example.description)[:MAX_DESCRIPTION_CHARS],
        )
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1024
        )
        was_training = model.training
        model.eval()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        if was_training:
            model.train()
        return self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if model is None or not self.examples:
            return control

        import copy
        import json
        import re

        predictions: list[set[str]] = []
        truths: list[set[str]] = []
        unparseable = 0
        for example in self.examples:
            raw = self._generate(model, example)
            match = re.search(r"\{.*\}", raw, re.S)
            skills: set[str] = set()
            if not match:
                unparseable += 1
            else:
                try:
                    payload = json.loads(match.group(0))
                    skills = {
                        str(s).strip().lower()
                        for s in payload.get("skills", [])
                        if str(s).strip()
                    }
                except Exception:  # noqa: BLE001 - malformed is a real result
                    unparseable += 1
            predictions.append(skills)
            truths.append({s.lower() for s in example.label})

        score = EpochScore(
            epoch=state.epoch or 0.0,
            micro_f1=_micro_f1(predictions, truths),
            empty_share=sum(1 for p in predictions if not p) / len(predictions),
            unparseable=unparseable,
            eval_loss=(metrics or {}).get("eval_loss"),
        )
        self.history.append(score)
        log.info(score.summary())

        if score.micro_f1 > self.best_f1:
            self.best_f1 = score.micro_f1
            self.best_epoch = score.epoch
            self._best_state = copy.deepcopy(
                {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                    if "lora" in k
                }
            )
            log.info("new best F1 at epoch %.0f", score.epoch)
        return control

    def restore_best(self, model) -> bool:
        """Put the best-F1 adapter weights back before saving."""
        if not self._best_state:
            return False
        model.load_state_dict(self._best_state, strict=False)
        log.info(
            "restored adapter from epoch %.0f (F1 %.3f)", self.best_epoch, self.best_f1
        )
        return True
