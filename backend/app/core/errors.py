"""Structured application and AI provider error classes."""

from __future__ import annotations


class AppError(Exception):
    """Base application error serialized into {error: {code, message}} without stack traces."""

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class DocumentError(AppError):
    """Raised on invalid, duplicate, oversized, corrupted, or scanned PDF uploads."""


class ChatError(AppError):
    """Raised on invalid chat requests or missing conversations."""


class LLMProviderError(AppError):
    """Raised when the LLM or embedding provider fails (missing key, 401/403, 429, 404, timeout)."""
