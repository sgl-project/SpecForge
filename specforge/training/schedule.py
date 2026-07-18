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


def resolve_online_total_steps(
    *,
    num_prompts: int,
    prompt_epochs: int,
    dp_size: int,
    batch_size: int,
    accumulation_steps: int,
) -> int:
    """Resolve the optimizer horizon for one finite online prompt plan.

    The online distributor emits only complete global optimizer windows.  Its
    tail policy therefore matches integer division by the exact dispatch
    quantum used here.
    """
    values = {
        "num_prompts": num_prompts,
        "prompt_epochs": prompt_epochs,
        "dp_size": dp_size,
        "batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
    }
    invalid = {name: value for name, value in values.items() if int(value) < 1}
    if invalid:
        raise ValueError(
            "online schedule inputs must be positive integers, got " f"{invalid}"
        )

    total_prompts = int(num_prompts) * int(prompt_epochs)
    global_optimizer_quantum = int(dp_size) * int(batch_size) * int(accumulation_steps)
    total_steps = total_prompts // global_optimizer_quantum
    if total_steps < 1:
        raise ValueError(
            "online prompt plan produces no optimizer step: "
            f"prompts={num_prompts}, prompt_epochs={prompt_epochs}, "
            f"dp_size={dp_size}, batch_size={batch_size}, "
            f"accumulation_steps={accumulation_steps}"
        )
    return total_steps


def validate_fixed_accumulation_plan(
    *,
    num_samples: int,
    batch_size: int,
    accumulation_steps: int,
    num_epochs: int,
    max_steps: Optional[int],
) -> None:
    """Reject a known partial accumulation before model/optimizer assembly.

    Fixed-ref loaders drop an incomplete sample batch. Gradient accumulation,
    however, spans epochs. If natural exhaustion would leave a partial
    optimizer window, training cannot commit that work durably; detect it up
    front unless an explicit ``max_steps`` cap stops at an earlier complete
    boundary.
    """
    micro_batches = (int(num_samples) // int(batch_size)) * int(num_epochs)
    complete_steps, remainder = divmod(micro_batches, int(accumulation_steps))
    stops_before_remainder = max_steps is not None and int(max_steps) <= complete_steps
    if remainder and not stops_before_remainder:
        raise ValueError(
            "fixed training plan ends with incomplete gradient accumulation: "
            f"{micro_batches} micro-batches across {num_epochs} epoch(s) is not "
            f"divisible by accumulation_steps={accumulation_steps} "
            f"(remainder={remainder}); adjust batch/accumulation/epochs or set "
            f"max_steps <= {complete_steps}"
        )


__all__ = [
    "resolve_online_total_steps",
    "resolve_total_steps",
    "validate_fixed_accumulation_plan",
]
