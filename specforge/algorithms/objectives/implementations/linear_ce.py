"""Fused linear projection and cross entropy implementation."""

import inspect
from functools import lru_cache
from importlib import metadata

import torch
from packaging.requirements import Requirement

from ..projection import LinearProjection
from .ce import projection_ce_support
from .registry import GroupResult, register_implementation


@lru_cache
def _linear_kernel():
    """Return the compatible Liger kernel class, or None if unavailable.

    The result is cached. Installed versions outside the project requirement
    raise before importing the kernel. Missing packages or a missing output
    capability return None; the caller reports that fused linear CE is unavailable.
    """
    from specforge.modeling.draft.dflash_kernels import LIGER_REQUIREMENT

    requirement = Requirement(LIGER_REQUIREMENT)
    try:
        installed = metadata.version(requirement.name)
    except metadata.PackageNotFoundError:
        return None
    if installed not in requirement.specifier:
        raise ValueError(
            f"fused linear CE requires {LIGER_REQUIREMENT}; installed version is {installed}"
        )
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
    if p is None or not isinstance(p.projection, LinearProjection):
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
        raise ValueError(
            "fused linear CE requires Liger with predicted-token output; "
            "install the supported Liger dependency or disable fused loss"
        )
    return None


@register_implementation(
    "linear_ce", supports=_linear_support, priority=10, losses=("ce",)
)
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
