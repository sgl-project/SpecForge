"""Local fused-kernel bindings for DFlash draft models."""

from __future__ import annotations

import inspect
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Callable

_LIGER_VERSION = "0.8.0"
_VALID_BACKENDS = frozenset({"liger", "torch"})


@lru_cache(maxsize=1)
def _load_liger_draft_forwards() -> tuple[Callable, Callable]:
    try:
        installed_version = version("liger-kernel")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "draft_kernel_backend='liger' requires " "`pip install 'specforge[liger]'`"
        ) from exc
    if installed_version != _LIGER_VERSION:
        raise RuntimeError(
            "draft_kernel_backend='liger' requires liger-kernel=="
            f"{_LIGER_VERSION}, found {installed_version}"
        )

    try:
        from liger_kernel.transformers.rms_norm import LigerRMSNorm
        from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "liger-kernel is installed but its RMSNorm/SwiGLU operators are "
            "unavailable; reinstall `specforge[liger]`"
        ) from exc

    rms_parameters = tuple(inspect.signature(LigerRMSNorm.forward).parameters)
    mlp_parameters = tuple(inspect.signature(LigerSwiGLUMLP.forward).parameters)
    if rms_parameters != ("self", "hidden_states") or mlp_parameters != (
        "self",
        "x",
    ):
        raise RuntimeError(
            "unsupported Liger DFlash kernel ABI: expected RMSNorm.forward"
            "(self, hidden_states) and SwiGLU.forward(self, x)"
        )
    return LigerRMSNorm.forward, LigerSwiGLUMLP.forward


def validate_dflash_draft_kernel_backend(backend: str) -> None:
    """Validate a draft-kernel backend before model construction or training."""

    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"draft_kernel_backend={backend!r}; must be one of "
            f"{sorted(_VALID_BACKENDS)}"
        )
    if backend == "liger":
        _load_liger_draft_forwards()


def configure_dflash_draft_kernels(draft_model, backend: str) -> None:
    """Bind fused forwards only on one DFlash draft model instance."""

    validate_dflash_draft_kernel_backend(backend)
    if backend == "torch":
        draft_model.draft_kernel_backend = backend
        return

    rms_forward, mlp_forward = _load_liger_draft_forwards()
    norms = [draft_model.norm, draft_model.hidden_norm]
    for layer in draft_model.layers:
        layer.mlp.forward = mlp_forward.__get__(layer.mlp, type(layer.mlp))
        norms.extend(
            (
                layer.input_layernorm,
                layer.post_attention_layernorm,
                layer.self_attn.q_norm,
                layer.self_attn.k_norm,
            )
        )

    for norm in norms:
        norm.offset = 0.0
        norm.casting_mode = "llama"
        # Residual paths share upstream gradients, so the in-place mode is unsafe.
        norm.in_place = False
        norm.row_mode = None
        norm.forward = rms_forward.__get__(norm, type(norm))
    draft_model.draft_kernel_backend = backend


__all__ = [
    "configure_dflash_draft_kernels",
    "validate_dflash_draft_kernel_backend",
]
