"""One LLM interface, two backends, every call logged.

Ollama is the backend that runs here: a local Llama needs no key and costs
nothing, which means the eval suite can run in CI on a laptop. Anthropic is
the backend that would run in production. Both implement `complete()`, so
Phase 6's fine-tuned model becomes a third entry and nothing above this file
changes.

Structured output is handled by asking for JSON, validating against a
pydantic model, and retrying with the validation error appended. Models do
not reliably return valid JSON on the first try, and a feature that crashes
when they do not is a demo, not a product. The retry count is logged, because
a rising average is the earliest signal that a prompt has regressed.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from joblens import db, observability
from joblens.config import get_settings
from joblens.llm.prompts import Prompt

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# USD per million tokens, input and output, from the public pricing page.
PRICING = {
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}


class BackendUnavailable(RuntimeError):
    """The configured LLM backend could not be reached.

    Ollama not running, the Anthropic API timing out, DNS failing inside a
    container: none of these are bugs in the request, and none should come
    back to a user as a 500 with a stack trace. The API maps this to 503.
    """


@dataclass
class Completion:
    text: str
    model: str
    backend: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    attempts: int = 1
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class OllamaBackend:
    """A local model over Ollama's HTTP API."""

    model: str
    base_url: str
    timeout: float
    name: str = "ollama"

    def complete(self, prompt: str, max_tokens: int, json_mode: bool) -> Completion:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        began = time.perf_counter()
        with httpx.Client(timeout=self.timeout) as http:
            response = http.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            body = response.json()
        return Completion(
            text=body.get("response", ""),
            model=self.model,
            backend=self.name,
            prompt_tokens=body.get("prompt_eval_count", 0),
            completion_tokens=body.get("eval_count", 0),
            latency_ms=int((time.perf_counter() - began) * 1000),
            cost_usd=0.0,  # runs on hardware already paid for
        )


@dataclass
class AnthropicBackend:
    """The Claude Messages API."""

    model: str
    api_key: str
    timeout: float
    name: str = "anthropic"

    def complete(self, prompt: str, max_tokens: int, json_mode: bool) -> Completion:
        began = time.perf_counter()
        with httpx.Client(timeout=self.timeout) as http:
            response = http.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
            body = response.json()
        usage = body.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        rate_in, rate_out = PRICING.get(self.model, (0.0, 0.0))
        return Completion(
            text="".join(block.get("text", "") for block in body.get("content", [])),
            model=self.model,
            backend=self.name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=int((time.perf_counter() - began) * 1000),
            cost_usd=(
                prompt_tokens / 1_000_000 * rate_in
                + completion_tokens / 1_000_000 * rate_out
            ),
        )


def get_backend():
    settings = get_settings()
    if settings.llm_backend == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "LLM_BACKEND=anthropic but ANTHROPIC_API_KEY is not set. "
                "Set the key, or use LLM_BACKEND=ollama to run locally."
            )
        return AnthropicBackend(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            timeout=settings.llm_timeout,
        )
    if settings.llm_backend == "ollama":
        return OllamaBackend(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            timeout=settings.llm_timeout,
        )
    raise ValueError(
        f"unknown LLM_BACKEND {settings.llm_backend!r}. known: ollama, anthropic"
    )


def _log_call(
    *,
    feature: str,
    completion: Completion | None,
    prompt_text: str,
    prompt: Prompt | None,
    ok: bool,
    attempts: int,
    error: str | None,
) -> None:
    """Record the call. Never let logging break the request it describes."""
    observability.record_llm_call(
        feature=feature,
        backend=completion.backend if completion else "unknown",
        model=completion.model if completion else "unknown",
        ok=ok,
        cost_usd=completion.cost_usd if completion else 0.0,
        latency_ms=completion.latency_ms if completion else None,
    )
    try:
        with db.connect() as conn:
            conn.execute(
                """
                insert into llm_calls
                    (feature, backend, model, prompt_name, prompt_version,
                     prompt, response, prompt_tokens, completion_tokens,
                     cost_usd, latency_ms, ok, attempts, error)
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    feature,
                    completion.backend if completion else "unknown",
                    completion.model if completion else "unknown",
                    prompt.name if prompt else None,
                    prompt.version if prompt else None,
                    prompt_text,
                    completion.text if completion else None,
                    completion.prompt_tokens if completion else None,
                    completion.completion_tokens if completion else None,
                    completion.cost_usd if completion else 0,
                    completion.latency_ms if completion else None,
                    ok,
                    attempts,
                    error,
                ),
            )
            conn.commit()
    except Exception:  # noqa: BLE001 - observability must not break the feature
        log.exception("could not log the LLM call")


def _raise_if_unavailable(backend, exc: Exception) -> None:
    """Translate a transport-level failure into BackendUnavailable.

    Shared by `complete` and `complete_structured` so that /chat and /match
    fail the same way: a 503 with a reason, never a 500 with a stack trace.
    """
    if isinstance(exc, httpx.TransportError):
        raise BackendUnavailable(f"{backend.name} backend unreachable: {exc}") from exc
    if isinstance(exc, httpx.HTTPStatusError):
        # Ollama answers 500 when the model fails to load, usually for
        # lack of memory. That is the backend's problem, not the caller's.
        raise BackendUnavailable(
            f"{backend.name} backend returned HTTP "
            f"{exc.response.status_code}: {exc.response.text[:200]}"
        ) from exc


def complete(
    prompt_text: str,
    *,
    feature: str,
    prompt: Prompt | None = None,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> Completion:
    """One free-text call, logged."""
    settings = get_settings()
    backend = get_backend()
    try:
        result = backend.complete(
            prompt_text, max_tokens or settings.llm_max_tokens, json_mode
        )
    except Exception as exc:
        _log_call(
            feature=feature,
            completion=None,
            prompt_text=prompt_text,
            prompt=prompt,
            ok=False,
            attempts=1,
            error=str(exc),
        )
        _raise_if_unavailable(backend, exc)
        raise
    _log_call(
        feature=feature,
        completion=result,
        prompt_text=prompt_text,
        prompt=prompt,
        ok=True,
        attempts=1,
        error=None,
    )
    return result


# Models wrap JSON in prose or fences however clearly you ask them not to.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> str:
    """Pull the JSON object out of a response that may be wrapped in prose."""
    fenced = _FENCE.search(text)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text.strip()


@dataclass
class StructuredResult:
    value: BaseModel
    completion: Completion
    attempts: int = 1
    repairs: list[str] = field(default_factory=list)


def complete_structured(
    prompt_text: str,
    schema: type[T],
    *,
    feature: str,
    prompt: Prompt | None = None,
    max_tokens: int | None = None,
    attempts: int = 3,
) -> StructuredResult:
    """Call, parse, validate, and retry with the error until it validates.

    The retry appends the actual validation error to the prompt rather than
    simply asking again. "Retry" on its own usually produces the same broken
    output; "field X must be an integer between 0 and 100, you sent 'high'"
    usually does not.
    """
    settings = get_settings()
    backend = get_backend()
    message = prompt_text
    repairs: list[str] = []
    last_error: Exception | None = None
    result: Completion | None = None

    for attempt in range(1, attempts + 1):
        try:
            result = backend.complete(
                message, max_tokens or settings.llm_max_tokens, True
            )
            value = schema.model_validate_json(extract_json(result.text))
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            repairs.append(str(exc)[:400])
            log.warning("structured output failed on attempt %s: %s", attempt, exc)
            message = (
                f"{prompt_text}\n\n"
                f"Your previous reply could not be parsed. The error was:\n"
                f"{str(exc)[:600]}\n\n"
                f"Reply with valid JSON only, matching the schema exactly. "
                f"No prose, no code fences."
            )
            continue
        except Exception as exc:
            _log_call(
                feature=feature,
                completion=None,
                prompt_text=message,
                prompt=prompt,
                ok=False,
                attempts=attempt,
                error=str(exc),
            )
            _raise_if_unavailable(backend, exc)
            raise

        result.attempts = attempt
        _log_call(
            feature=feature,
            completion=result,
            prompt_text=message,
            prompt=prompt,
            ok=True,
            attempts=attempt,
            error="; ".join(repairs) or None,
        )
        return StructuredResult(
            value=value, completion=result, attempts=attempt, repairs=repairs
        )

    _log_call(
        feature=feature,
        completion=result,
        prompt_text=message,
        prompt=prompt,
        ok=False,
        attempts=attempts,
        error=str(last_error),
    )
    raise ValueError(
        f"{feature}: no valid {schema.__name__} after {attempts} attempts. "
        f"last error: {last_error}"
    )


def spend(days: int = 7) -> list[dict]:
    """What the LLM features have cost, by day and feature."""
    with db.connect() as conn:
        return conn.execute(
            """
            select date_trunc('day', created_at)::date as day,
                   feature, backend, count(*) as calls,
                   sum(prompt_tokens + completion_tokens) as tokens,
                   round(sum(cost_usd), 4) as usd,
                   round(avg(latency_ms)) as avg_ms,
                   round(avg(attempts), 2) as avg_attempts,
                   count(*) filter (where not ok) as failures
              from llm_calls
             where created_at > now() - make_interval(days => %s)
             group by 1, 2, 3
             order by 1 desc, calls desc
            """,
            (days,),
        ).fetchall()
