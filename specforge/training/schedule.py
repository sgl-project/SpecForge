"""Training-step horizon shared by optimizer and strategy schedules."""

from __future__ import annotations

from typing import Optional


def resolve_total_steps(
    *,
    total_steps: Optional[int],
    max_steps: Optional[int],
    num_samples: Optional[int],
    batch_size: int,
    accumulation_steps: int,
    num_epochs: int,
) -> int:
    """Resolve one optimizer-step horizon from explicit limits or finite data."""
    if total_steps is not None:
        return total_steps
    if max_steps is not None:
        return max_steps
    if num_samples is None:
        raise ValueError(
            "a streaming training run requires training.total_steps or "
            "training.max_steps so optimizer and loss schedules share a horizon"
        )

    micro_batches_per_epoch = num_samples // batch_size
    optimizer_steps = (micro_batches_per_epoch * num_epochs) // accumulation_steps
    if optimizer_steps < 1:
        raise ValueError(
            "training data produces no optimizer step: "
            f"samples={num_samples}, batch_size={batch_size}, "
            f"accumulation_steps={accumulation_steps}, num_epochs={num_epochs}"
        )
    return optimizer_steps


__all__ = ["resolve_total_steps"]
