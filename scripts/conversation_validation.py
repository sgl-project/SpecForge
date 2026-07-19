"""Shared conversation validation helpers for data regeneration scripts."""

from typing import Any


def has_think_marker(content: str) -> bool:
    lowered = content.lower()
    return "<think>" in lowered or "</think>" in lowered


def validate_conversation(
    messages: Any,
    *,
    error_style: str = "validation",
) -> str | None:
    if not isinstance(messages, list) or not messages:
        if error_style == "regeneration":
            return "Missing or empty conversations list"
        return "conversations must be a non-empty list"

    expected_role = "user"
    saw_user = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            if error_style == "regeneration":
                return f"Invalid message at position {index}: expected object"
            return f"message {index} must be an object"
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            if error_style == "regeneration":
                return (
                    f"Invalid message content at position {index}: "
                    "expected non-empty string"
                )
            return f"message {index} content must be a non-empty string"

        if role == "system" and not saw_user:
            continue
        if role not in {"user", "assistant"}:
            if error_style == "regeneration":
                return f"Invalid message role: {role}"
            return f"message {index} has unsupported role {role!r}"
        if role != expected_role:
            if error_style == "regeneration":
                if not saw_user and role == "assistant":
                    return "Data starts with an assistant message"
                return (
                    f"Invalid conversation role order at position {index}: "
                    f"expected {expected_role}, got {role}"
                )
            return f"message {index} expected role {expected_role!r}, got {role!r}"

        saw_user = True
        expected_role = "assistant" if role == "user" else "user"

    if not saw_user:
        if error_style == "regeneration":
            return "Data does not contain a user message"
        return "conversation has no user message"
    return None
