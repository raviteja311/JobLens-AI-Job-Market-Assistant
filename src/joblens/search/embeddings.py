"""Embedding backends behind one interface.

Two of them, because the interesting question is not "does semantic search
work" but "what does it cost". A local MiniLM runs on the machine already
paid for and returns 384 dimensions. An API model returns better vectors and
bills per token. Both implement the same three members, so the eval harness
can score them against the same golden set and the comparison is apples to
apples.

Every call records latency and token count. Cost is the model's published
rate times the tokens, which is an estimate and is labelled as one.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from joblens.config import get_settings

log = logging.getLogger(__name__)

LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_DIMENSION = 384

# USD per million tokens, from the provider's public pricing page. An estimate
# for reporting, not a billing record.
API_COST_PER_MTOK = {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13}


@dataclass
class Usage:
    """What a batch of embedding cost us."""

    calls: int = 0
    texts: int = 0
    tokens: int = 0
    seconds: float = 0.0
    usd: float = 0.0

    def add(self, *, texts: int, tokens: int, seconds: float, usd: float) -> None:
        self.calls += 1
        self.texts += texts
        self.tokens += tokens
        self.seconds += seconds
        self.usd += usd

    @property
    def ms_per_text(self) -> float:
        return (self.seconds * 1000 / self.texts) if self.texts else 0.0


class Embedder(Protocol):
    name: str
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class LocalEmbedder:
    """sentence-transformers on the CPU. No key, no bill, no network."""

    model_name: str = LOCAL_MODEL
    batch_size: int = 32
    usage: Usage = field(default_factory=Usage)
    _model: object | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return self.model_name.rsplit("/", 1)[-1]

    @property
    def dimension(self) -> int:
        return LOCAL_DIMENSION

    def _load(self):
        if self._model is None:
            # Imported here rather than at module scope: it drags in torch,
            # which costs about two seconds and 300MB of RSS, and the ingest
            # pipeline has no reason to pay that.
            from sentence_transformers import SentenceTransformer

            log.info("loading %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        model = self._load()
        started = time.perf_counter()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        elapsed = time.perf_counter() - started
        # No tokeniser bill to report, but the count is what makes the local
        # and API rows of the comparison table line up.
        tokens = sum(len(t) // 4 for t in texts)
        self.usage.add(texts=len(texts), tokens=tokens, seconds=elapsed, usd=0.0)
        return np.asarray(vectors, dtype=np.float32)


@dataclass
class ApiEmbedder:
    """An OpenAI-compatible embeddings endpoint.

    Kept behind the same interface so switching is a config flag, and so the
    eval harness can score it without knowing which one it has.
    """

    model_name: str = "text-embedding-3-small"
    dimension_override: int | None = None
    batch_size: int = 128
    usage: Usage = field(default_factory=Usage)

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def dimension(self) -> int:
        return self.dimension_override or 1536

    def encode(self, texts: list[str]) -> np.ndarray:
        import httpx

        settings = get_settings()
        if not settings.embedding_api_key:
            raise RuntimeError(
                "EMBEDDING_API_KEY is not set, so the API embedder cannot run. "
                "Use --embedder local, or set the key to reproduce the "
                "local-vs-API comparison."
            )
        vectors: list[list[float]] = []
        with httpx.Client(timeout=settings.request_timeout) as http:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                began = time.perf_counter()
                response = http.post(
                    f"{settings.embedding_api_base}/embeddings",
                    headers={"Authorization": f"Bearer {settings.embedding_api_key}"},
                    json={"model": self.model_name, "input": batch},
                )
                response.raise_for_status()
                payload = response.json()
                elapsed = time.perf_counter() - began
                tokens = payload.get("usage", {}).get("total_tokens", 0)
                rate = API_COST_PER_MTOK.get(self.model_name, 0.0)
                self.usage.add(
                    texts=len(batch),
                    tokens=tokens,
                    seconds=elapsed,
                    usd=tokens / 1_000_000 * rate,
                )
                vectors.extend(item["embedding"] for item in payload["data"])
        matrix = np.asarray(vectors, dtype=np.float32)
        # Normalised here so cosine similarity is a dot product everywhere,
        # matching what the local backend already returns.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-9, None)


def get_embedder(name: str = "local") -> Embedder:
    if name == "local":
        return LocalEmbedder()
    if name == "api":
        return ApiEmbedder()
    raise ValueError(f"unknown embedder {name!r}. known: local, api")
