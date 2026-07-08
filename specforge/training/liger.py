"""Optional Liger-kernel integration for the DFlash draft model."""

from __future__ import annotations

from typing import Any, Callable


def _load_liger_apply() -> Callable[..., Any]:
    """Resolve Liger lazily so the base installation has no dependency on it."""

    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3
    except ModuleNotFoundError as exc:
        if exc.name in {"liger_kernel", "liger_kernel.transformers"}:
            raise ImportError(
                "model.use_liger_kernel=true requires the optional "
                "`specforge[liger]` extra. Install it with "
                "`pip install \"specforge[liger]\"`."
            ) from exc
        raise
    return apply_liger_kernel_to_qwen3


def maybe_apply_liger_kernel(cfg: Any) -> None:
    """Patch Qwen3 classes before constructing a Liger-enabled DFlash draft.

    The option is deliberately limited to the DFlash strategy in this PR.
    Other DFlash-family strategies may opt in later with their own validation
    and coverage rather than inheriting a global Transformers monkey-patch.
    """

    if not cfg.model.use_liger_kernel:
        return
    if cfg.training.strategy != "dflash":
        raise ValueError(
            "model.use_liger_kernel is currently supported only with "
            "training.strategy=dflash"
        )

    apply_liger_kernel_to_qwen3 = _load_liger_apply()
    # Rope is excluded because DFlash owns its asymmetric rotary embedding.
    # Cross-entropy kernels are excluded because DFlash computes its loss
    # outside Transformers' Qwen3 forward.
    apply_liger_kernel_to_qwen3(
        rope=False,
        rms_norm=True,
        swiglu=True,
        cross_entropy=False,
        fused_linear_cross_entropy=False,
    )
