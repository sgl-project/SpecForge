"""Loss-mask predicates shared by online and offline data paths."""

from collections.abc import Sequence
from itertools import pairwise
from typing import Any


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _unwrap_1d(loss_mask: Any) -> Sequence[Any]:
    values = (
        loss_mask
        if _is_sequence(loss_mask)
        else loss_mask.tolist() if hasattr(loss_mask, "tolist") else list(loss_mask)
    )
    if not _is_sequence(values):
        raise ValueError("expected a 1D or batch-1 2D loss mask")
    if values and _is_sequence(values[0]):
        if len(values) != 1:
            raise ValueError("expected one loss-mask row")
        values = values[0]
    if values and _is_sequence(values[0]):
        raise ValueError("expected a 1D or batch-1 2D loss mask")
    return values


def normalize_loss_mask_1d(loss_mask: Any) -> list[int]:
    """Unwrap one loss mask (1D or batch-1 2D) to a 1D list of ints."""

    values = _unwrap_1d(loss_mask)
    normalized: list[int] = []
    for item in values:
        if _is_sequence(item):
            raise ValueError("expected a 1D or batch-1 2D loss mask")
        normalized.append(int(item))
    return normalized


def has_consecutive_supervised_tokens(loss_mask: Any) -> bool:
    """Return whether one sample contains adjacent supervised tokens."""

    values = _unwrap_1d(loss_mask)
    return any(
        bool(current) and bool(following) for current, following in pairwise(values)
    )


__all__ = ["has_consecutive_supervised_tokens", "normalize_loss_mask_1d"]
