"""Extensible LLMProvider abstraction with production Google Gemini implementation."""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

from google import genai
from google.genai import types

from app.core.config import get_settings
from app.core.errors import LLMProviderError

logger = logging.getLogger("documind.llm")

_KEY_REDACT_RE = re.compile(r"AIza[0-9A-Za-z_\-]{10,}")
_FAILOVER_ELIGIBLE_CODES = {
    "RATE_LIMITED",
    "MODEL_UNAVAILABLE",
    "SERVICE_UNAVAILABLE",
    "TIMEOUT",
    "PROVIDER_ERROR",
    "NETWORK_ERROR",
    "EMPTY_RESPONSE",
}


def _sanitize_error_detail(exc: Exception) -> str:
    raw = _KEY_REDACT_RE.sub("[REDACTED]", str(exc))
    return raw.replace("\n", " ").strip()[:240]


@dataclass(frozen=True)
class HistoryTurn:
    role: str  # "user" or "model"
    text: str


class LLMProvider(ABC):
    """Provider abstraction allowing seamless swaps to OpenAI, Anthropic, or local models."""

    provider_name: str
    chat_model: str
    embedding_model: str

    @abstractmethod
    def is_configured(self) -> bool:
        ...

    @abstractmethod
    def generate_text(
        self,
        *,
        user_prompt: str,
        system_instruction: str | None = None,
        history: list[HistoryTurn] | None = None,
    ) -> str:
        ...

    @abstractmethod
    def stream_text(
        self,
        *,
        user_prompt: str,
        system_instruction: str | None = None,
        history: list[HistoryTurn] | None = None,
    ) -> Iterator[str]:
        ...

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


def classify_provider_exception(exc: Exception, model_name: str = "") -> LLMProviderError:
    if isinstance(exc, LLMProviderError):
        return exc

    raw = str(exc)
    lower = raw.lower()
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None) or 0

    if (
        "api key not valid" in lower
        or "api_key_invalid" in lower
        or "permission_denied" in lower
        or "unauthorized" in lower
        or status in (401, 403)
    ):
        return LLMProviderError(
            code="INVALID_CREDENTIALS",
            message=(
                "The configured Gemini API key was rejected by the provider. "
                "Verify GEMINI_API_KEY in your environment or Settings > Secrets."
            ),
            status_code=401,
        )

    if (
        "resource_exhausted" in lower
        or "quota" in lower
        or "rate limit" in lower
        or status == 429
    ):
        return LLMProviderError(
            code="RATE_LIMITED",
            message=(
                "The Gemini API rate limit or quota was exceeded. "
                "Please wait a moment or check your billing/quota settings."
            ),
            status_code=429,
        )

    if "not_found" in lower or "not found" in lower or status == 404:
        model_hint = f" ('{model_name}')" if model_name else ""
        return LLMProviderError(
            code="MODEL_UNAVAILABLE",
            message=(
                f"Configured Gemini model{model_hint} is unavailable or unsupported for this API key. "
                "Update GEMINI_MODEL or GEMINI_EMBEDDING_MODEL in your .env file."
            ),
            status_code=404,
        )

    if (
        "unavailable" in lower
        or "overloaded" in lower
        or "high demand" in lower
        or status == 503
    ):
        return LLMProviderError(
            code="SERVICE_UNAVAILABLE",
            message=(
                "The AI model is temporarily experiencing high demand. Please try again in a moment."
            ),
            status_code=503,
        )

    if "timeout" in lower or "timed out" in lower or "deadline" in lower or status == 504:
        return LLMProviderError(
            code="TIMEOUT",
            message="The AI provider timed out before returning a response. Please try again.",
            status_code=504,
        )

    if (
        "connection" in lower
        or "network" in lower
        or "dns" in lower
        or "connecterror" in lower
    ):
        return LLMProviderError(
            code="NETWORK_ERROR",
            message="Unable to reach the Gemini API due to a network error. Please verify internet connectivity.",
            status_code=503,
        )

    return LLMProviderError(
        code="PROVIDER_ERROR",
        message="The AI provider encountered an unexpected error while processing your request. Please try again.",
        status_code=502,
    )


class GeminiProvider(LLMProvider):
    """Google Gemini API provider powered by the official google-genai Python SDK."""

    provider_name = "Google Gemini"

    def __init__(self) -> None:
        settings = get_settings()
        self.chat_model = settings.gemini_model
        self.fallback_chat_model = settings.gemini_fallback_model.strip()
        self.embedding_model = settings.gemini_embedding_model
        self.timeout_seconds = settings.provider_timeout_seconds
        self._model_cooldown_until: dict[str, float] = {}
        self._client_cache: dict[tuple[str, int], genai.Client] = {}

    def is_configured(self) -> bool:
        return bool(get_settings().effective_gemini_api_key)

    def _create_client(self, timeout_seconds: float | None = None) -> genai.Client:
        api_key = get_settings().effective_gemini_api_key
        if not api_key:
            raise LLMProviderError(
                code="MISSING_API_KEY",
                message=(
                    "GEMINI_API_KEY is not configured on the FastAPI backend. "
                    "Set GEMINI_API_KEY in backend/.env (or Settings > Secrets) to enable live AI generation and embeddings."
                ),
                status_code=503,
            )
        effective_timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        timeout_ms = max(1000, int(effective_timeout * 1000))
        cache_key = (api_key, timeout_ms)
        cached = self._client_cache.get(cache_key)
        if cached is not None:
            return cached
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=timeout_ms,
                headers={"User-Agent": "aistudio-build"},
            ),
        )
        if len(self._client_cache) > 8:
            self._client_cache.clear()
        self._client_cache[cache_key] = client
        return client

    @staticmethod
    def _build_contents(
        user_prompt: str, history: list[HistoryTurn] | None
    ) -> list[types.Content]:
        contents: list[types.Content] = []
        for turn in history or []:
            contents.append(
                types.Content(
                    role="user" if turn.role == "user" else "model",
                    parts=[types.Part.from_text(text=turn.text)],
                )
            )
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])
        )
        return contents

    @staticmethod
    def _build_generate_config(
        system_instruction: str | None, model_name: str
    ) -> types.GenerateContentConfig:
        kwargs: dict = {}
        if system_instruction:
            kwargs["system_instruction"] = system_instruction
        if model_name == "gemini-3.1-flash-lite":
            try:
                kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.MINIMAL
                )
            except Exception:
                pass
        elif model_name.startswith("gemini-4-") or model_name.startswith("gemini-3"):
            try:
                kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.LOW
                )
            except Exception:
                pass
        return types.GenerateContentConfig(**kwargs)

    def _candidate_chat_models(self) -> list[str]:
        models = [self.chat_model]
        for entry in self.fallback_chat_model.split(","):
            candidate = entry.strip()
            if candidate and candidate not in models:
                models.append(candidate)
        if len(models) <= 1:
            return models
        now = time.monotonic()
        active = [m for m in models if self._model_cooldown_until.get(m, 0.0) <= now]
        # Skip models in active 429/404 cooldown as long as at least one healthy model remains;
        # otherwise fall back to trying the configured chain so recovery happens automatically.
        return active if active else models

    def _mark_model_cooldown(self, model_name: str, code: str) -> None:
        if code in {"RATE_LIMITED", "MODEL_UNAVAILABLE"}:
            self._model_cooldown_until[model_name] = time.monotonic() + 120.0

    def _with_retry(self, fn, model_name: str, max_attempts: int = 2):
        last_exc: Exception | None = None
        backoff_schedule = (0.8, 1.8)
        for attempt in range(1, max_attempts + 1):
            try:
                return fn()
            except Exception as exc:
                classified = classify_provider_exception(exc, model_name=model_name)
                logger.info(
                    "Gemini call failed model=%s attempt=%d/%d code=%s type=%s detail=%s",
                    model_name,
                    attempt,
                    max_attempts,
                    classified.code,
                    type(exc).__name__,
                    _sanitize_error_detail(exc),
                )
                last_exc = classified
                if classified.code in {
                    "MISSING_API_KEY",
                    "INVALID_CREDENTIALS",
                    "MODEL_UNAVAILABLE",
                }:
                    raise classified from exc
                if attempt < max_attempts:
                    delay = backoff_schedule[min(attempt - 1, len(backoff_schedule) - 1)]
                    time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _prefer_actionable_error(
        prior: LLMProviderError | None, current: LLMProviderError
    ) -> LLMProviderError:
        if (
            prior is not None
            and current.code == "MODEL_UNAVAILABLE"
            and prior.code in {"RATE_LIMITED", "SERVICE_UNAVAILABLE", "TIMEOUT"}
        ):
            return prior
        return current

    def generate_text(
        self,
        *,
        user_prompt: str,
        system_instruction: str | None = None,
        history: list[HistoryTurn] | None = None,
    ) -> str:
        contents = self._build_contents(user_prompt, history)
        models = self._candidate_chat_models()
        last_err: LLMProviderError | None = None

        for idx, model_name in enumerate(models):
            has_fallback = idx < len(models) - 1
            attempt_timeout = min(self.timeout_seconds, 20.0) if has_fallback else min(self.timeout_seconds, 45.0)
            client = self._create_client(timeout_seconds=attempt_timeout)
            config = self._build_generate_config(system_instruction, model_name)

            def _call(active_client=client, active_model: str = model_name, active_cfg=config) -> str:
                response = active_client.models.generate_content(
                    model=active_model,
                    contents=contents,
                    config=active_cfg,
                )
                text = (response.text or "").strip()
                if not text:
                    raise LLMProviderError(
                        code="EMPTY_RESPONSE",
                        message="The AI model returned an empty response. Please try rephrasing your question.",
                        status_code=502,
                    )
                return text

            try:
                return self._with_retry(
                    _call,
                    model_name=model_name,
                    max_attempts=1 if has_fallback else 2,
                )
            except LLMProviderError as err:
                last_err = self._prefer_actionable_error(last_err, err)
                if has_fallback and err.code in _FAILOVER_ELIGIBLE_CODES:
                    self._mark_model_cooldown(model_name, err.code)
                    logger.info(
                        "Failing over Gemini chat generation from %s (%s) to %s",
                        model_name,
                        err.code,
                        models[idx + 1],
                    )
                    continue
                logger.warning(
                    "Gemini chat generation failed on %s (%s): %s",
                    model_name,
                    last_err.code,
                    last_err.message,
                )
                raise last_err from err

        assert last_err is not None
        raise last_err

    def stream_text(
        self,
        *,
        user_prompt: str,
        system_instruction: str | None = None,
        history: list[HistoryTurn] | None = None,
    ) -> Iterator[str]:
        contents = self._build_contents(user_prompt, history)
        models = self._candidate_chat_models()
        last_err: LLMProviderError | None = None

        for idx, model_name in enumerate(models):
            has_fallback = idx < len(models) - 1
            attempt_timeout = min(self.timeout_seconds, 20.0) if has_fallback else min(self.timeout_seconds, 45.0)
            max_attempts = 1 if has_fallback else 2
            client = self._create_client(timeout_seconds=attempt_timeout)
            config = self._build_generate_config(system_instruction, model_name)

            for attempt in range(1, max_attempts + 1):
                emitted = False
                try:
                    stream = client.models.generate_content_stream(
                        model=model_name,
                        contents=contents,
                        config=config,
                    )
                    for chunk in stream:
                        piece = getattr(chunk, "text", None)
                        if piece:
                            emitted = True
                            yield piece
                    if not emitted:
                        raise LLMProviderError(
                            code="EMPTY_RESPONSE",
                            message="The AI response stream finished without content. Please try again.",
                            status_code=502,
                        )
                    return
                except Exception as exc:
                    classified = classify_provider_exception(exc, model_name=model_name)
                    logger.info(
                        "Gemini stream attempt failed model=%s attempt=%d/%d emitted=%s code=%s type=%s detail=%s",
                        model_name,
                        attempt,
                        max_attempts,
                        emitted,
                        classified.code,
                        type(exc).__name__,
                        _sanitize_error_detail(exc),
                    )
                    last_err = self._prefer_actionable_error(last_err, classified)
                    if emitted or classified.code in {
                        "MISSING_API_KEY",
                        "INVALID_CREDENTIALS",
                    }:
                        raise classified from exc
                    if has_fallback and classified.code in _FAILOVER_ELIGIBLE_CODES:
                        self._mark_model_cooldown(model_name, classified.code)
                        logger.info(
                            "Failing over Gemini stream from %s (%s) to %s",
                            model_name,
                            classified.code,
                            models[idx + 1],
                        )
                        break
                    # On TIMEOUT, skip repeating another slow streaming attempt and try unary directly.
                    if (
                        attempt < max_attempts
                        and classified.code not in {"MODEL_UNAVAILABLE", "TIMEOUT"}
                    ):
                        time.sleep(0.8)
                        continue
                    # Final resilience check on the last candidate: try unary generate_content
                    # once if streaming failed pre-token due to transient 503/500/timeout/empty.
                    if not has_fallback and classified.code in {
                        "SERVICE_UNAVAILABLE",
                        "PROVIDER_ERROR",
                        "TIMEOUT",
                        "EMPTY_RESPONSE",
                    }:
                        try:
                            unary_client = self._create_client(
                                timeout_seconds=min(self.timeout_seconds, 45.0)
                            )
                            unary_response = unary_client.models.generate_content(
                                model=model_name,
                                contents=contents,
                                config=config,
                            )
                            unary_text = (unary_response.text or "").strip()
                            if unary_text:
                                yield unary_text
                                return
                        except Exception as unary_exc:
                            unary_classified = classify_provider_exception(
                                unary_exc, model_name=model_name
                            )
                            logger.warning(
                                "Gemini unary fallback failed model=%s code=%s detail=%s",
                                model_name,
                                unary_classified.code,
                                _sanitize_error_detail(unary_exc),
                            )
                            last_err = self._prefer_actionable_error(last_err, unary_classified)
                    logger.warning(
                        "Gemini stream failed permanently on %s (%s): %s",
                        model_name,
                        last_err.code,
                        last_err.message,
                    )
                    raise last_err from exc

        if last_err is not None:
            raise last_err

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._create_client()
        # Wrap each text as its own Content object; in google-genai, gemini-embedding-2
        # converts a bare list[str] into a single multi-part Content (returning 1 vector),
        # whereas list[types.Content] produces one independent vector per chunk.
        content_items = [
            types.Content(parts=[types.Part.from_text(text=item)]) for item in texts
        ]

        def _call() -> list[list[float]]:
            result = client.models.embed_content(
                model=self.embedding_model,
                contents=content_items,
            )
            embeddings = [list(item.values) for item in (result.embeddings or [])]
            if len(embeddings) != len(texts) or any(len(vec) == 0 for vec in embeddings):
                raise LLMProviderError(
                    code="EMBEDDING_ERROR",
                    message="The embedding API returned incomplete vector embeddings.",
                    status_code=502,
                )
            return embeddings

        return self._with_retry(_call, model_name=self.embedding_model)


_provider_singleton: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    global _provider_singleton
    if _provider_singleton is None:
        _provider_singleton = GeminiProvider()
    return _provider_singleton
