"""Loss-mask predicates shared by online and offline data paths."""

from collections.abc import Sequence
from typing import Any


def has_consecutive_supervised_tokens(loss_mask: Any) -> bool:
    """Return whether one sample contains adjacent supervised tokens."""

    values = loss_mask.tolist() if hasattr(loss_mask, "tolist") else list(loss_mask)
    if values and isinstance(values[0], Sequence):
        if len(values) != 1:
            raise ValueError("expected one loss-mask row")
        values = list(values[0])
    return any(
        bool(current) and bool(following)
        for current, following in zip(values, values[1:])
    )


__all__ = ["has_consecutive_supervised_tokens"]
