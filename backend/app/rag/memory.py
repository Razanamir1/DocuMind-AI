"""Token-aware conversation memory windowing, follow-up contextualization, and title generation."""

from __future__ import annotations

import math
import re
from typing import Sequence

from app.providers.llm import HistoryTurn

FOLLOW_UP_PATTERN = re.compile(
    r"\b(it|its|they|them|their|that|those|this|these|he|she|above|previous|more|also|what about|how about|and what|elaborate|compare|former|latter|same)\b",
    re.IGNORECASE,
)


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def select_conversation_window(
    messages: Sequence[tuple[str, str]],
    *,
    max_tokens: int = 3200,
    max_messages: int = 12,
) -> list[HistoryTurn]:
    """Selects recent (role, content) turns fitting within the token budget."""
    if not messages:
        return []

    recent = list(messages)[-max_messages:]
    selected: list[HistoryTurn] = []
    used_tokens = 0

    for role, content in reversed(recent):
        tokens = estimate_tokens(content)
        if selected and used_tokens + tokens > max_tokens:
            break
        selected.insert(
            0,
            HistoryTurn(role="user" if role == "user" else "model", text=content),
        )
        used_tokens += tokens

    return selected


def build_contextual_retrieval_query(
    current_question: str, prior_user_messages: Sequence[str]
) -> str:
    trimmed = current_question.strip()
    if not prior_user_messages:
        return trimmed

    looks_like_follow_up = len(trimmed) < 26 or bool(FOLLOW_UP_PATTERN.search(trimmed))
    if not looks_like_follow_up:
        return trimmed

    prior_context = prior_user_messages[-1].strip()[:180]
    return f"{trimmed} (context: {prior_context})"


def generate_conversation_title(first_message: str) -> str:
    cleaned = re.sub(r"^[#>*\-\s]+", "", re.sub(r"\s+", " ", first_message)).strip()
    if not cleaned:
        return "New Conversation"
    if len(cleaned) <= 52:
        return cleaned
    cut = cleaned[:52]
    last_space = cut.rfind(" ")
    return f"{(cut[:last_space] if last_space > 24 else cut).strip()}…"
