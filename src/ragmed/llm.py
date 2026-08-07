"""LLM client.

One interface, three consumers with very different reliability requirements:
generation (user-facing, streams), query rewriting (latency-critical, optional), and
the LLM-judge (must return parseable structure or the eval silently degrades).

The judge is the reason ``complete_json`` exists and is strict. A judge that returns
prose when it was asked for JSON, and gets silently coerced to a default score, is
worse than no judge - it produces a number that looks like a measurement. Here a
parse failure raises, and the caller records it as an error rather than a score.

``NullLLM`` is a first-class provider, not a stub: the entire retrieval half of the
eval suite is designed to run with no model available at all.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

import httpx

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMUnavailable(LLMError):
    """The backend is not reachable, or the requested model is not installed."""


class LLMParseError(LLMError):
    """The model did not return the structure it was asked for."""


@runtime_checkable
class LLM(Protocol):
    name: str

    def available(self) -> bool: ...

    def complete(self, prompt: str, system: str | None = None, **kw: Any) -> str: ...

    def complete_json(self, prompt: str, system: str | None = None, **kw: Any) -> Any: ...

    def stream(self, prompt: str, system: str | None = None, **kw: Any) -> Iterator[str]: ...


class NullLLM:
    """Refuses to pretend. Used when no backend is configured."""

    name = "null"

    def available(self) -> bool:
        return False

    def _fail(self) -> None:
        raise LLMUnavailable(
            "no LLM configured (generation.provider='null'). Retrieval metrics run "
            "without one; generation metrics require a backend."
        )

    def complete(self, prompt: str, system: str | None = None, **kw: Any) -> str:
        self._fail()
        raise AssertionError("unreachable")

    def complete_json(self, prompt: str, system: str | None = None, **kw: Any) -> Any:
        self._fail()
        raise AssertionError("unreachable")

    def stream(self, prompt: str, system: str | None = None, **kw: Any) -> Iterator[str]:
        self._fail()
        raise AssertionError("unreachable")


# Models sometimes wrap JSON in prose or a fenced block even when told not to.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse JSON from a model response, tolerating fences and surrounding prose.

    Tolerant about *packaging*, strict about *content*: if there is no parseable JSON
    value in the response this raises rather than inventing one.
    """
    text = text.strip()
    if not text:
        raise LLMParseError("empty response")

    candidates: list[str] = [text]
    fenced = _FENCE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    # Fall back to the outermost brace/bracket span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise LLMParseError(f"no parseable JSON in response: {text[:300]!r}")


class OllamaLLM:
    """Client for a local Ollama server (https://ollama.com).

    Ollama's ``format: json`` option constrains decoding to valid JSON, which removes
    most of the judge's failure modes at the source rather than repairing them after.
    """

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 800,
        num_ctx: int = 8192,
        timeout_s: float = 300.0,
        max_retries: int = 3,
        keep_alive: str = "30m",
    ):
        self.model = model
        self.name = f"ollama:{model}"
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        # Ollama evicts an idle model after 5 minutes by default. An eval sweep has
        # natural gaps (scoring, disk writes, the retrieval half of a run), so the
        # model gets unloaded and reloaded repeatedly - and a request that lands
        # mid-reload comes back as a 500 or an empty body. Holding it resident for
        # the length of a run removes that churn entirely.
        self.keep_alive = keep_alive
        self._client = httpx.Client(timeout=timeout_s)
        self._available: bool | None = None

    def close(self) -> None:
        self._client.close()

    def available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            resp = self._client.get(f"{self.host}/api/tags", timeout=5.0)
            resp.raise_for_status()
            installed = {m["name"] for m in resp.json().get("models", [])}
        except Exception as exc:  # noqa: BLE001
            log.warning("Ollama not reachable at %s: %s", self.host, exc)
            self._available = False
            return False

        # Ollama reports tagged names ("llama3.1:8b"); accept an untagged request.
        if self.model in installed or any(m.split(":")[0] == self.model for m in installed):
            self._available = True
        else:
            log.warning(
                "Ollama is running but model %r is not installed (have: %s). Run: ollama pull %s",
                self.model,
                ", ".join(sorted(installed)) or "none",
                self.model,
            )
            self._available = False
        return self._available

    def _messages(self, prompt: str, system: str | None) -> list[dict[str, str]]:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _options(self, **kw: Any) -> dict[str, Any]:
        return {
            "temperature": kw.get("temperature", self.temperature),
            "num_predict": kw.get("max_tokens", self.max_tokens),
            # Never omit this. Ollama's default is 4096 no matter what the model
            # supports, and over-long prompts are truncated silently - see
            # GenerationConfig.num_ctx for what that costs.
            "num_ctx": kw.get("num_ctx", self.num_ctx),
        }

    def complete(self, prompt: str, system: str | None = None, **kw: Any) -> str:
        payload = {
            "model": self.model,
            "messages": self._messages(prompt, system),
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": self._options(**kw),
        }
        if kw.get("json_mode"):
            payload["format"] = "json"

        # Small local models intermittently return an empty body or a 500 under
        # memory pressure - observed repeatedly on a 4GB card. These are transient
        # and a retry almost always succeeds, so they should not surface as an
        # eval-wide failure. A persistent one still raises.
        last: str = ""
        for attempt in range(self.max_retries):
            try:
                resp = self._client.post(f"{self.host}/api/chat", json=payload)
                if resp.status_code >= 500:
                    last = f"HTTP {resp.status_code}"
                else:
                    resp.raise_for_status()
                    content = resp.json().get("message", {}).get("content", "")
                    if content.strip():
                        return content
                    last = "empty response"
            except httpx.HTTPError as exc:
                last = str(exc)

            if attempt < self.max_retries - 1:
                backoff = 1.5 * (attempt + 1)
                log.warning(
                    "Ollama call failed (%s), retry %d/%d in %.1fs",
                    last, attempt + 1, self.max_retries, backoff,
                )
                time.sleep(backoff)

        raise LLMUnavailable(
            f"Ollama request failed after {self.max_retries} attempts: {last}"
        )

    def complete_json(self, prompt: str, system: str | None = None, **kw: Any) -> Any:
        # A *truncated* response ('{"question": "What') is non-empty, so it sails past
        # complete()'s retry and only fails at the parse step. Small models do this
        # intermittently on long prompts, and it is just as transient as an empty
        # body - so it gets its own retry rather than being reported as a hard error.
        last: LLMParseError | None = None
        for attempt in range(self.max_retries):
            raw = self.complete(prompt, system, json_mode=True, **kw)
            try:
                return extract_json(raw)
            except LLMParseError as exc:
                last = exc
                if attempt < self.max_retries - 1:
                    log.warning(
                        "unparseable JSON (attempt %d/%d): %.80s",
                        attempt + 1, self.max_retries, str(exc),
                    )
                    time.sleep(1.0 * (attempt + 1))
        assert last is not None
        raise last

    def stream(self, prompt: str, system: str | None = None, **kw: Any) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": self._messages(prompt, system),
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": self._options(**kw),
        }
        try:
            with self._client.stream("POST", f"{self.host}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    piece = event.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if event.get("done"):
                        break
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"Ollama stream failed: {exc}") from exc


def build_llm(provider: str, model: str, host: str, **kw: Any) -> LLM:
    if provider == "ollama":
        return OllamaLLM(model=model, host=host, **kw)
    if provider == "null":
        return NullLLM()
    raise ValueError(f"unknown LLM provider {provider!r}")


def generation_llm(cfg: Any) -> LLM:
    return build_llm(
        cfg.generation.provider,
        cfg.generation.model,
        cfg.generation.host,
        temperature=cfg.generation.temperature,
        max_tokens=cfg.generation.max_tokens,
        num_ctx=cfg.generation.num_ctx,
        timeout_s=cfg.generation.timeout_s,
    )


def judge_llm(cfg: Any) -> LLM:
    return build_llm(
        cfg.judge.provider,
        cfg.judge.model,
        cfg.judge.host,
        temperature=cfg.judge.temperature,
        # Judgements are short structured verdicts; a large budget only invites prose.
        max_tokens=512,
        num_ctx=cfg.judge.num_ctx,
        timeout_s=cfg.judge.timeout_s,
    )
