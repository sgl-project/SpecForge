"""Memory-efficient frozen-linear cross entropy with per-token gradients.

Liger computes the hidden-state gradient during its chunked forward without
retaining the full ``tokens x vocabulary`` logits tensor. Its public unreduced
backward assumes a scalar upstream gradient, so this adapter saves the
unweighted hidden gradient and applies DFlash's per-token weights itself.
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Callable

import torch

_LIGER_VERSION = "0.8.0"
_REQUIRED_FORWARD_PARAMETERS = {
    "_input",
    "weight",
    "target",
    "bias",
    "reduction",
    "return_token_accuracy",
    "return_predicted_tokens",
}


@lru_cache(maxsize=1)
def _load_liger_forward() -> Callable:
    try:
        installed_version = version("liger-kernel")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "linear_cross_entropy_backend='liger' requires "
            "`pip install 'specforge[liger]'`"
        ) from exc
    if installed_version != _LIGER_VERSION:
        raise RuntimeError(
            "linear_cross_entropy_backend='liger' requires liger-kernel=="
            f"{_LIGER_VERSION}, found {installed_version}"
        )

    try:
        from liger_kernel.ops.fused_linear_cross_entropy import (
            fused_linear_cross_entropy_forward,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "liger-kernel is installed but its fused linear cross entropy "
            "operator is unavailable; reinstall `specforge[liger]`"
        ) from exc

    parameters = set(inspect.signature(fused_linear_cross_entropy_forward).parameters)
    missing = _REQUIRED_FORWARD_PARAMETERS - parameters
    if missing:
        raise RuntimeError(
            "unsupported Liger fused linear cross entropy ABI; missing parameters: "
            + ", ".join(sorted(missing))
        )
    return fused_linear_cross_entropy_forward


def validate_liger_installation() -> None:
    """Fail before training when the pinned Liger operator is unavailable."""

    _load_liger_forward()


class _FrozenLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, target, bias):
        result = _load_liger_forward()(
            _input=hidden,
            weight=weight,
            target=target,
            bias=bias,
            reduction="none",
            return_token_accuracy=True,
        )
        if not isinstance(result, tuple) or len(result) != 7:
            result_count = len(result) if isinstance(result, tuple) else type(result)
            raise RuntimeError(
                "unsupported Liger fused linear cross entropy return ABI: "
                f"expected 7 values, got {result_count}"
            )

        (
            loss,
            _z_loss,
            token_accuracy,
            _predicted_tokens,
            grad_input,
            _grad_weight,
            _grad_bias,
        ) = result
        ctx.save_for_backward(grad_input.detach())
        ctx.mark_non_differentiable(token_accuracy)
        return loss, token_accuracy

    @staticmethod
    def backward(ctx, grad_loss, _grad_accuracy):
        (grad_input,) = ctx.saved_tensors
        if grad_loss is None:
            return torch.zeros_like(grad_input), None, None, None
        compute_dtype = torch.promote_types(grad_input.dtype, torch.float32)
        scaled = grad_input.to(compute_dtype) * grad_loss.reshape(-1, 1).to(
            compute_dtype
        )
        return scaled.to(grad_input.dtype), None, None, None


def frozen_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-token CE and correctness without materializing full logits."""

    if hidden.ndim != 2 or weight.ndim != 2 or target.ndim != 1:
        raise ValueError("expected hidden [N,H], weight [V,H], and target [N]")
    if hidden.shape[0] != target.shape[0] or hidden.shape[1] != weight.shape[1]:
        raise ValueError("hidden, weight, and target shapes are incompatible")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
        raise ValueError("bias shape must match the LM-head vocabulary dimension")
    if target.dtype != torch.long:
        raise ValueError("target must use torch.long token ids")
    if hidden.device != weight.device or hidden.device != target.device:
        raise ValueError("hidden, weight, and target must be on the same device")
    if bias is not None and bias.device != hidden.device:
        raise ValueError("bias must be on the same device as hidden")
    if hidden.dtype != weight.dtype:
        raise ValueError("hidden and weight must use the same dtype")
    if weight.requires_grad or (bias is not None and bias.requires_grad):
        raise ValueError("Liger DFlash linear cross entropy requires a frozen LM head")
    return _FrozenLinearCrossEntropy.apply(
        hidden.contiguous(), weight, target.contiguous(), bias
    )


__all__ = ["frozen_linear_cross_entropy", "validate_liger_installation"]
