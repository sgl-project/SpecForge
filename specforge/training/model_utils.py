"""Small model-assembly utilities shared by training strategies."""

from __future__ import annotations

from typing import Any, Optional


def resolve_mask_token_id(
    *,
    explicit: Optional[int],
    tokenizer: Any,
    embedding_vocab_size: int,
    draft_model: Any = None,
) -> int:
    """Resolve a MASK id without resizing the draft embedding table.

    Priority is explicit config, draft metadata, tokenizer MASK id, an unused
    tokenizer slot in the existing embedding table, then a standard fallback
    token. Every candidate is range checked before it can reach a model lookup.
    """

    method_config = getattr(
        getattr(draft_model, "config", None), "dflash_config", None
    ) or {}
    candidates = [
        ("explicit mask token", explicit),
        ("draft config mask token", method_config.get("mask_token_id")),
        ("tokenizer mask token", getattr(tokenizer, "mask_token_id", None)),
    ]
    try:
        tokenizer_size = len(tokenizer)
    except TypeError:
        tokenizer_size = embedding_vocab_size
    if tokenizer_size < embedding_vocab_size:
        candidates.append(("unused embedding slot", tokenizer_size))
    candidates.extend(
        (name, getattr(tokenizer, attribute, None))
        for name, attribute in (
            ("tokenizer pad token", "pad_token_id"),
            ("tokenizer eos token", "eos_token_id"),
            ("tokenizer unknown token", "unk_token_id"),
        )
    )

    for source, candidate in candidates:
        if candidate is None:
            continue
        candidate = int(candidate)
        if not 0 <= candidate < embedding_vocab_size:
            if source.startswith("explicit"):
                raise ValueError(
                    f"mask token id {candidate} is outside draft embedding "
                    f"vocabulary [0, {embedding_vocab_size})"
                )
            continue
        return candidate
    raise ValueError(
        "unable to resolve a mask token id within the draft embedding "
        f"vocabulary [0, {embedding_vocab_size})"
    )


__all__ = ["resolve_mask_token_id"]
