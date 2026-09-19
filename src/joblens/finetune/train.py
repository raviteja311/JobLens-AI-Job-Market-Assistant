"""LoRA fine-tuning of a small model for skill extraction.

The task is narrow on purpose. "Read a job posting, return a JSON list of the
technologies it names" is exactly the shape of problem where a 0.5B model
should be able to match an 8B one after seeing a few hundred examples: the
reasoning is shallow, the output format is rigid, and the vocabulary is
closed. If distillation does not work here it does not work anywhere in this
project, which is what makes it worth measuring rather than assuming.

Why LoRA rather than full fine-tuning. Full fine-tuning updates all 494M
parameters and needs optimiser state for every one of them, which is roughly
6GB in fp32 before activations. LoRA freezes the base weights and trains two
small matrices per attention projection, so the trainable count here is about
1% of the model and it fits on a laptop CPU. It also means the artifact is a
few megabytes of adapter rather than a gigabyte of weights.

Rank 16 with alpha 32. Rank is the width of the bottleneck and therefore how
much the adapter can change the model. A rank-4 adapter is enough to teach a
model an output format; a rank-64 one has the capacity to learn new facts and
the capacity to overfit 300 examples. 16 is the middle, and the rank sweep is
in docs/experiments.md rather than being asserted here.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from joblens.finetune import dataset
from joblens.finetune.callbacks import SkillF1Callback

log = logging.getLogger(__name__)

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "skill-lora"

# Long enough for a 1,800 character posting plus the instructions and the
# answer. Anything longer is truncated, which loses the tail of a few long
# postings and is cheaper than padding every batch to the worst case.
MAX_SEQ_LENGTH = 1024


@dataclass
class TrainConfig:
    base_model: str = BASE_MODEL
    output_dir: Path = ADAPTER_DIR
    epochs: float = 3.0
    learning_rate: float = 2e-4
    batch_size: int = 1
    # Batch size 1 on CPU, so the gradient is noisy. Accumulating 8 steps
    # gives an effective batch of 8 without the memory of one.
    gradient_accumulation: int = 8
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_seq_length: int = MAX_SEQ_LENGTH
    seed: int = 42
    eval_fraction: float = 0.1
    # How many held-out postings the F1 callback generates on each epoch.
    # Generation is slow on CPU, so this is small: it exists to catch a
    # collapse, not to measure quality. The frozen test set does that.
    validation_slice: int = 16


@dataclass
class TrainResult:
    config: TrainConfig
    train_examples: int
    eval_examples: int
    seconds: float
    train_loss: float
    eval_loss: float
    trainable_params: int
    total_params: int
    adapter_path: Path
    best_f1: float = -1.0
    history: list[dict] = field(default_factory=list)
    f1_history: list = field(default_factory=list)

    def summary(self) -> str:
        share = self.trainable_params / self.total_params * 100
        return (
            f"rank {self.config.lora_rank}: trained {self.train_examples} examples "
            f"in {self.seconds / 60:.1f} min, "
            f"train loss {self.train_loss:.4f}, eval loss {self.eval_loss:.4f}, "
            f"best micro F1 {self.best_f1:.3f}, "
            f"{self.trainable_params:,} of {self.total_params:,} params "
            f"({share:.2f}%)"
        )

    def f1_table(self) -> str:
        lines = [
            "| epoch | micro F1 | empty | unparseable | eval loss |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
        for s in self.f1_history:
            loss = f"{s.eval_loss:.4f}" if s.eval_loss is not None else "n/a"
            lines.append(
                f"| {s.epoch:.0f} | {s.micro_f1:.3f} | {s.empty_share:.0%} "
                f"| {s.unparseable} | {loss} |"
            )
        return "\n".join(lines)


def verify_masking(trainer, tokenizer) -> dict:
    """Assert the prompt really is masked, and say so in numbers.

    The first run recorded `completion_only_loss: true` in its training log
    for a run where nothing at all was masked. A flag that is set is not a
    flag that works, so this reads the first batch back and refuses to train
    if the mask is missing.
    """
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"][0]
    masked = int((labels == -100).sum())
    scored = len(labels) - masked
    share = masked / len(labels) if len(labels) else 0.0
    if share < 0.5:
        raise RuntimeError(
            f"only {masked} of {len(labels)} positions are masked ({share:.0%}). "
            "The loss is being computed on the prompt, which is the bug that "
            "collapsed the first run. Check assistant_only_loss and the chat "
            "template's generation markers."
        )
    log.info(
        "loss mask: %s of %s positions masked (%.0f%%), %s scored",
        masked,
        len(labels),
        share * 100,
        scored,
    )
    return {"length": len(labels), "masked": masked, "scored": scored}


def build_dataset(examples: list[dataset.Example], config: TrainConfig):
    """Turn Examples into the chat-format dataset trl expects.

    Returns the tokenised splits plus the raw held-out Examples, because the
    F1 callback needs the original title and description to generate from
    and a tokenised row cannot give it those.
    """
    import random

    from datasets import Dataset

    shuffled = list(examples)
    random.Random(config.seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * config.eval_fraction))
    held_out, kept = shuffled[:cut], shuffled[cut:]

    train_data = Dataset.from_list([dataset.to_chat(e) for e in kept])
    eval_data = Dataset.from_list([dataset.to_chat(e) for e in held_out])
    return train_data, eval_data, held_out


def train(
    examples: list[dataset.Example], config: TrainConfig | None = None
) -> TrainResult:
    """One LoRA run. Returns the numbers, writes the adapter."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    config = config or TrainConfig()
    if len(examples) < 50:
        raise ValueError(
            f"fine-tuning needs at least 50 examples, got {len(examples)}. "
            "Run `python -m joblens distil-label` first."
        )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.base_model, dtype=torch.float32)
    model.config.use_cache = False

    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # The attention projections only. Adapting the MLP as well roughly
        # doubles the trainable count for a task that is about output format
        # rather than new knowledge.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("trainable %s of %s parameters", f"{trainable:,}", f"{total:,}")

    train_data, eval_data, held_out = build_dataset(examples, config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    args = SFTConfig(
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        # trl 1.13's SFTConfig has warmup_steps but not warmup_ratio. Two
        # steps out of roughly seventy is the same 3% the ratio would give.
        warmup_steps=2,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        max_length=config.max_seq_length,
        seed=config.seed,
        report_to=[],
        # Loss on the answer only.
        #
        # This must be assistant_only_loss, not completion_only_loss. The
        # first run set completion_only_loss=True, trl ignored it without a
        # warning because that flag only applies to prompt-completion
        # datasets and this one is conversational, and every one of the 316
        # positions in a training example kept its loss. The completion is
        # about 8 tokens, so 97% of the gradient went on reciting the
        # instructions and the job posting, including the schema example.
        # The model learned to emit an empty list and nothing else.
        #
        # assistant_only_loss needs {% generation %} markers in the chat
        # template. Qwen2.5 has them. `verify_masking` below asserts the
        # mask is real rather than trusting the flag a second time.
        assistant_only_loss=True,
        bf16=False,
        fp16=False,
    )

    # Scored on generated output every epoch, because the first run's eval
    # loss fell all the way down to a model with a micro F1 of zero.
    f1_callback = SkillF1Callback(
        examples=held_out[: config.validation_slice], tokenizer=tokenizer
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        callbacks=[f1_callback],
    )

    masking = verify_masking(trainer, tokenizer)

    began = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - began

    history = [row for row in trainer.state.log_history]
    train_loss = next(
        (r["loss"] for r in reversed(history) if "loss" in r), float("nan")
    )
    eval_loss = next(
        (r["eval_loss"] for r in reversed(history) if "eval_loss" in r), float("nan")
    )

    # Keep the best-F1 epoch, not the last one and not the best loss.
    f1_callback.restore_best(trainer.model)
    trainer.model.save_pretrained(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    (config.output_dir / "training.json").write_text(
        json.dumps(
            {
                "base_model": config.base_model,
                "examples": len(examples),
                "epochs": config.epochs,
                "lora_rank": config.lora_rank,
                "lora_alpha": config.lora_alpha,
                "learning_rate": config.learning_rate,
                "seconds": round(elapsed, 1),
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "trainable_params": trainable,
                "total_params": total,
                "assistant_only_loss": True,
                "loss_mask": masking,
                "split_hash": dataset.split_hash(examples),
                "best_f1": f1_callback.best_f1,
                "best_epoch": f1_callback.best_epoch,
                "f1_by_epoch": [
                    {
                        "epoch": s.epoch,
                        "micro_f1": s.micro_f1,
                        "empty_share": s.empty_share,
                        "unparseable": s.unparseable,
                        "eval_loss": s.eval_loss,
                    }
                    for s in f1_callback.history
                ],
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return TrainResult(
        f1_history=f1_callback.history,
        best_f1=f1_callback.best_f1,
        config=config,
        train_examples=len(train_data),
        eval_examples=len(eval_data),
        seconds=elapsed,
        train_loss=train_loss,
        eval_loss=eval_loss,
        trainable_params=trainable,
        total_params=total,
        adapter_path=config.output_dir,
        history=history,
    )
