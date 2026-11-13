"""Execution implementations selected by projection and loss contracts."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from ..losses import LossContext, TokenLoss
from ..predictions import Predictions

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TermInputs:
    """Resolved inputs for one loss term.

    Attributes:
        loss: Mathematical loss and its options; None for already-computed auxiliary terms.
        mask: Supervised positions, shaped [units, positions].
        weights: Effective per-position weights, aligned with mask.
        denominator: Scalar used after additive chunk statistics are merged.
        context: Teacher-mass and normalization inputs for contextual losses.
    """

    loss: TokenLoss | None
    mask: torch.Tensor
    weights: torch.Tensor
    denominator: torch.Tensor
    context: LossContext


@dataclass
class GroupResult:
    """Unnormalized output of an execution implementation.

    Attributes:
        numerators: Term indices mapped to weighted scalar loss sums. Values
            must not include the term coefficient or division by its denominator.
        predictions: Optional output names mapped to predicted token IDs,
            shaped [units, positions], for metric computation.
        statistics: Term indices mapped to reusable named tensors. The executor
            detaches these before passing them to auxiliary callbacks.
    """

    numerators: dict[int, torch.Tensor]
    predictions: dict[str, torch.Tensor] = field(default_factory=dict)
    statistics: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionRequest:
    """Data shared by applicability checks and kernel execution.

    Attributes:
        predictions: Deferred projection, aligned inputs and the loss plan.
        terms: Resolved inputs in the same order as LossPlan.terms.
        labels: Token targets shaped [units, positions].
        teacher: Optional teacher logits aligned with the selected vocabulary.
        outputs: Zero-argument callable that lazily projects and caches named
            outputs within this evaluation. Only compute callbacks may call it.
        materialized: The same output cache; applicability checks may inspect
            whether it is populated without triggering projection.
    """

    predictions: Predictions
    terms: tuple[TermInputs, ...]
    labels: torch.Tensor
    teacher: torch.Tensor | None
    outputs: Callable
    materialized: dict


@dataclass(frozen=True)
class Implementation:
    """A registered optimization and its applicability contract.

    Attributes:
        name: Unique identifier, also used to break equal-priority ties.
        priority: Selection priority; larger values are considered first.
        supports: Callable(request, group) returning None when applicable,
            otherwise a human-readable rejection reason. It must not project.
        compute: Callable(request, group) returning a GroupResult. Runtime
            failures propagate; they are not treated as applicability failures.
    """

    name: str
    priority: int
    supports: Callable
    compute: Callable


_IMPLEMENTATIONS = {}


def register_implementation(name, *, supports, priority=0):
    """Register an optimized implementation for one or more loss terms.

    Args:
        name: Unique implementation identifier.
        supports: Callable(request, group) returning None if supported or a
            rejection-reason string. It checks capabilities without projection.
        priority: Selection priority; higher values take precedence.

    Returns:
        A decorator that registers and returns the compute callable unchanged.
        The callable accepts an ExecutionRequest and a tuple of term indices.

    Raises:
        ValueError: The decorated implementation's name is already registered.
    """

    def register(compute):
        if name in _IMPLEMENTATIONS:
            raise ValueError(f"Execution implementation {name!r} is already registered")
        _IMPLEMENTATIONS[name] = Implementation(name, priority, supports, compute)
        return compute

    return register


def select_implementation(request, group, *, name="auto"):
    """Select an applicable implementation without executing its kernel.

    Args:
        request: Projection, resolved term inputs and lazy output access.
        group: Tuple of indices into request.predictions.plan.terms to execute
            together. A one-element tuple requests independent execution.
        name: "auto" checks priority order, breaking ties by ascending name.
            Any other value requires that specific registered implementation.

    Returns:
        The first applicable Implementation, or None when auto mode finds no
        match. The caller decides whether to split the group or use reference
        execution. Applicability checks may load optional kernel dependencies.

    Raises:
        ValueError: An explicitly named implementation is unknown or rejects
            the request. Exceptions raised by applicability checks propagate.
    """
    from . import domino_ce, linear_ce, token_kernels  # noqa: F401

    candidates = sorted(
        _IMPLEMENTATIONS.values(), key=lambda item: (-item.priority, item.name)
    )
    if name != "auto":
        if name not in _IMPLEMENTATIONS:
            raise ValueError(f"Unknown execution implementation: {name!r}")
        candidates = [_IMPLEMENTATIONS[name]]
    for entry in candidates:
        reason = entry.supports(request, group)
        if reason is None:
            logger.debug("Loss group %s uses %s", group, entry.name)
            return entry
        logger.debug("Loss group %s skips %s: %s", group, entry.name, reason)
        if name != "auto":
            raise ValueError(f"{name} cannot execute this loss group: {reason}")
    logger.debug("Loss group %s uses reference execution", group)
    return None
