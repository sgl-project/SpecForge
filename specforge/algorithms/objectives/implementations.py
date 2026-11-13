"""Execution implementations selected by projection and loss contracts."""

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

import torch

from .losses import LossContext, TokenLoss
from .predictions import Predictions
from .projection import LinearProjection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TermInputs:
    """Resolved inputs for one loss term.

    Attributes:
        loss: Mathematical loss and its options.
        mask: Supervised positions, shaped [units, positions].
        weights: Effective per-position weights, aligned with mask.
        denominator: Scalar used after additive chunk statistics are merged.
        context: Teacher-mass and normalization inputs for contextual losses.
    """

    loss: TokenLoss
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
    """

    numerators: dict[int, torch.Tensor]
    predictions: dict[str, torch.Tensor] = field(default_factory=dict)


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
    from . import token_kernels  # noqa: F401

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


def projection_ce_support(request, group):
    """Return None or a rejection reason for the requested fused CE group."""
    p = request.predictions
    if not p.hidden.is_cuda or p.hidden.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        return "requires CUDA FP16/BF16/FP32"
    if p.hidden.numel() == 0:
        return "empty projection uses the reference path"
    if not p.projection.chunk_safe or not p.projection.recompute_safe:
        return "projection does not permit chunked/recomputed fusion"
    if request.materialized:
        return "projection is already materialized"
    if p.auxiliary is not None:
        return "auxiliary computation consumes full projection outputs"
    if p.vocab_mask is not None:
        return "vocabulary mapping requires a different projection implementation"
    if any(
        request.terms[i].loss.kind != "ce" or request.terms[i].loss.options
        for i in group
    ):
        return "requires ordinary hard-label CE"
    return None


@lru_cache
def _linear_kernel():
    """Return the compatible Liger kernel class, or None if unavailable.

    The result is cached. Missing optional packages and an unsupported kernel
    signature disable this optimization; unrelated import errors propagate.
    """
    try:
        from liger_kernel.ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] in ("liger_kernel", "triton"):
            return None
        raise
    # Older kernels cannot provide accuracy without materializing logits.
    if (
        "return_predicted_tokens"
        not in inspect.signature(
            LigerFusedLinearCrossEntropyFunction.forward
        ).parameters
    ):
        return None
    return LigerFusedLinearCrossEntropyFunction


def _linear_support(request, group):
    """Return None or a reason the requested linear CE cannot be fused."""
    p = request.predictions
    if not isinstance(p.projection, LinearProjection):
        return "requires an explicit linear projection"
    if (
        len(group) != 1
        or p.plan.terms[group[0]].output != "logits"
        or p.projection_inputs
    ):
        return "requires one ordinary linear output"
    reason = projection_ce_support(request, group)
    if reason:
        return reason
    from torch import nn

    from specforge.modeling.draft.bindings import BoundLMHead

    head = p.projection.head
    if type(head) not in (nn.Linear, BoundLMHead):
        return "wrapped or transformed heads use their normal forward"
    if head._forward_hooks or head._forward_pre_hooks:
        return "head hooks require normal module execution"
    if type(head.weight) not in (torch.Tensor, nn.Parameter) or head.weight.ndim != 2:
        return "requires an unsharded matrix"
    if head.weight.device != p.hidden.device or head.weight.dtype != p.hidden.dtype:
        return "requires matching activation and weight device/dtype"
    if _linear_kernel() is None:
        return "Liger linear CE with predicted-token output is unavailable"
    return None


@register_implementation("linear_ce", supports=_linear_support, priority=10)
def linear_ce(request, group):
    """Fuse a plain linear projection with hard-label cross entropy.

    Args:
        request: Validated LinearProjection, token labels and position weights.
        group: One term index naming the logits output.

    Returns:
        GroupResult with a weighted CE sum and predicted token IDs, without
        retaining the full vocabulary logits. Supports must be checked first.
    """
    p = request.predictions
    weight, bias = p.projection.weight_and_bias()
    losses, _, _, predicted = _linear_kernel().apply(
        p.hidden.flatten(0, 1),
        weight,
        request.labels.flatten(),
        bias,
        None,
        -100,
        0.0,
        0.0,
        "none",
        None,
        False,
        None,
        False,
        False,
        True,
    )
    index = group[0]
    return GroupResult(
        {
            index: (
                losses.reshape_as(request.labels) * request.terms[index].weights
            ).sum()
        },
        {"logits": predicted.reshape_as(request.labels)},
    )
