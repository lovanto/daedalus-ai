from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import OrderedDict
from typing import AsyncIterator

import httpx

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LRU cache for identical prompt → response pairs (saves Ollama round-trips)
# ---------------------------------------------------------------------------

class _LRUCache:
    """Thread-safe in-memory LRU cache backed by OrderedDict."""

    def __init__(self, maxsize: int = 100) -> None:
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._maxsize = maxsize

    def _key(self, prompt: str, model: str, system: str, temperature: float) -> str:
        raw = f"{model}|{system}|{temperature:.2f}|{prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, prompt: str, model: str, system: str, temperature: float) -> str | None:
        k = self._key(prompt, model, system, temperature)
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]
        return None

    def set(self, prompt: str, model: str, system: str, temperature: float, value: str) -> None:
        k = self._key(prompt, model, system, temperature)
        if k in self._cache:
            self._cache.move_to_end(k)
        self._cache[k] = value
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)


_cache = _LRUCache(maxsize=100)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class OllamaError(Exception):
    pass


class OllamaTimeoutError(OllamaError):
    """Raised when Ollama does not respond within the configured timeout.
    Unlike connection errors, timeouts are not retried — the model is busy."""
    pass


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OllamaClient:
    def __init__(
        self,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.default_model = default_model or settings.OLLAMA_MODEL
        self.timeout = timeout if timeout is not None else settings.OLLAMA_TIMEOUT
        self.max_retries = max_retries if max_retries is not None else settings.OLLAMA_MAX_RETRIES
        self._mode = settings.OLLAMA_MODE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.7,
    ) -> str:
        """Single-turn text generation with caching and retry."""
        resolved_model = model or self.default_model
        resolved_system = system or ""

        cached = _cache.get(prompt, resolved_model, resolved_system, temperature)
        if cached is not None:
            logger.debug("cache hit for prompt hash")
            return cached

        if self._mode == "cloud":
            result = await self._cloud_generate(prompt, model=resolved_model, system=resolved_system)
        else:
            result = await self._retry(self._local_generate, prompt,
                                       model=resolved_model, system=resolved_system,
                                       temperature=temperature)

        _cache.set(prompt, resolved_model, resolved_system, temperature, result)
        return result

    async def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        system: str | None = None,
    ) -> str:
        """Multi-turn chat completion with retry."""
        if self._mode == "cloud":
            return await self._cloud_chat(messages, model=model, system=system)
        return await self._retry(self._local_chat, messages, model=model, system=system)

    async def stream_generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Streaming generation (local mode only — no caching)."""
        payload: dict = {
            "model": model or self.default_model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                async with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        if token := chunk.get("response"):
                            yield token
                        if chunk.get("done"):
                            break
            except httpx.HTTPError as exc:
                raise OllamaError(f"stream failed: {exc}") from exc

    async def list_models(self) -> list[str]:
        """Return names of locally available Ollama models."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                return [m["name"] for m in resp.json().get("models", [])]
        except httpx.HTTPError:
            return []

    async def is_reachable(self) -> tuple[bool, list[str]]:
        models = await self.list_models()
        return bool(models), models

    # ------------------------------------------------------------------
    # Local (Ollama) implementation
    # ------------------------------------------------------------------

    async def _local_generate(self, prompt: str, *, model: str, system: str, temperature: float) -> str:
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
            "think": self._mode == "local" and settings.OLLAMA_THINK,
        }
        if system:
            payload["system"] = system
        data = await self._post("/api/generate", payload)
        return data.get("response", "")

    async def _local_chat(self, messages: list[dict], *, model: str | None, system: str | None) -> str:
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)
        payload = {
            "model": model or self.default_model,
            "messages": all_messages,
            "stream": False,
            "think": settings.OLLAMA_THINK,
        }
        data = await self._post("/api/chat", payload)
        return data.get("message", {}).get("content", "")

    async def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()
        except httpx.ConnectError as exc:
            raise OllamaError(
                f"Ollama is not reachable at {self.base_url}. "
                "Make sure `ollama serve` is running."
            ) from exc
        except httpx.TimeoutException as exc:
            limit = f"{self.timeout}s" if self.timeout is not None else "no limit set"
            raise OllamaTimeoutError(f"Ollama request timed out ({limit}).") from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaError(
                f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc

    # ------------------------------------------------------------------
    # Cloud (Anthropic) implementation
    # ------------------------------------------------------------------

    async def _cloud_generate(self, prompt: str, *, model: str, system: str) -> str:
        """Route to Anthropic Messages API when OLLAMA_MODE=cloud."""
        if not settings.ANTHROPIC_API_KEY:
            raise OllamaError("ANTHROPIC_API_KEY is not set — required for cloud mode")

        payload = {
            "model": settings.ANTHROPIC_MODEL,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system

        headers = {
            "x-api-key": settings.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["content"][0]["text"]
        except httpx.HTTPError as exc:
            raise OllamaError(f"Anthropic API error: {exc}") from exc

    async def _cloud_chat(
        self, messages: list[dict], *, model: str | None, system: str | None
    ) -> str:
        # Flatten to a single generate call via the last user message
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return await self._cloud_generate(last, model=model or self.default_model, system=system or "")

    # ------------------------------------------------------------------
    # Retry with exponential backoff
    # ------------------------------------------------------------------

    async def _retry(self, fn, *args, **kwargs) -> str:
        """Call fn with exponential backoff, up to self.max_retries attempts.
        Timeouts are not retried — they indicate a busy/slow model, not a transient error."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return await fn(*args, **kwargs)
            except OllamaTimeoutError:
                raise  # never retry timeouts
            except OllamaError as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt  # 1s, 2s, 4s …
                    logger.warning(
                        "Ollama call failed (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1, self.max_retries, wait, exc,
                    )
                    await asyncio.sleep(wait)
        raise last_exc  # type: ignore[misc]


# Module-level singleton
ollama = OllamaClient()
