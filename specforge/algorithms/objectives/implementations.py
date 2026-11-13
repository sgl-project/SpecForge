"""Execution implementations selected by projection and loss contracts."""

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

import torch

from .losses import LossContext, TokenLoss
from .predictions import Predictions
from .projection import CorrectedProjection, LinearProjection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TermInputs:
    loss: TokenLoss
    mask: torch.Tensor
    weights: torch.Tensor
    denominator: torch.Tensor
    context: LossContext


@dataclass
class GroupResult:
    numerators: dict[int, torch.Tensor]
    predictions: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionRequest:
    predictions: Predictions
    terms: tuple[TermInputs, ...]
    labels: torch.Tensor
    teacher: torch.Tensor | None
    outputs: Callable
    materialized: dict


@dataclass(frozen=True)
class Implementation:
    name: str
    priority: int
    supports: Callable
    compute: Callable


_IMPLEMENTATIONS = {}


def register_implementation(name, *, supports, priority=0):
    """supports returns None when applicable, otherwise a fallback reason."""

    def register(compute):
        if name in _IMPLEMENTATIONS:
            raise ValueError(f"Execution implementation {name!r} is already registered")
        _IMPLEMENTATIONS[name] = Implementation(name, priority, supports, compute)
        return compute

    return register


def select_implementation(request, group, *, name="auto"):
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


def _ce_support(request, group):
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


def _corrected_support(request, group):
    p = request.predictions
    if not isinstance(p.projection, CorrectedProjection):
        return "requires a compact corrected projection"
    if len(group) != 2 or {p.plan.terms[i].output for i in group} != {
        "logits",
        "base_logits",
    }:
        return "requires the base/final CE pair"
    if any(
        p.plan.terms[i].mask is not None or p.plan.terms[i].weighting is not None
        for i in group
    ):
        return "joint CE requires shared supervision weights"
    return _ce_support(request, group)


@register_implementation("corrected_ce", supports=_corrected_support, priority=20)
def corrected_ce(request, group):
    from specforge.core.domino_loss import domino_weighted_cross_entropy

    p, projection = request.predictions, request.predictions.projection
    base, correction = projection.project_parts(p.hidden, **p.projection_inputs)
    final, original, predicted, _ = domino_weighted_cross_entropy(
        base.flatten(0, 1),
        correction.reshape(-1, correction.shape[-1]),
        request.labels.flatten(),
        request.terms[group[0]].weights.flatten(),
        projection.block_size,
        projection.suffix_start,
        use_fused=True,
    )
    values = {"logits": final, "base_logits": original}
    return GroupResult(
        {i: values[p.plan.terms[i].output] for i in group},
        {"logits": predicted.reshape_as(request.labels)},
    )


@lru_cache
def _linear_kernel():
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
    p = request.predictions
    if not isinstance(p.projection, LinearProjection):
        return "requires an explicit linear projection"
    if (
        len(group) != 1
        or p.plan.terms[group[0]].output != "logits"
        or p.projection_inputs
    ):
        return "requires one ordinary linear output"
    reason = _ce_support(request, group)
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
