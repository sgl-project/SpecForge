"""Environment defaults that must be installed before PyTorch compilation."""

from __future__ import annotations

import os

_INDUCTOR_GEMM_BACKENDS = "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"


def configure_flex_attention_inductor(attention_backend: str) -> bool:
    """Keep an ATen fallback for dynamic FlexAttention shapes.

    Some otherwise-valid decoding shapes have no Triton autotune candidate.
    PyTorch raises ``NoValidChoicesError`` unless ATen remains available as a
    fallback.  Preserve an explicit operator override, but make the safe
    ``ATEN,TRITON`` setting the default for SpecForge FlexAttention training.

    Returns ``True`` when this call installed the default.
    """
    if attention_backend != "flex_attention" or _INDUCTOR_GEMM_BACKENDS in os.environ:
        return False
    os.environ[_INDUCTOR_GEMM_BACKENDS] = "ATEN,TRITON"
    # Config may already be imported indirectly by Transformers or Accelerate;
    # environment variables are read only once when that module initializes.
    # Update the live value as well so a late programmatic entry is just as
    # reliable as a fresh CLI process.
    from torch._inductor import config as inductor_config

    inductor_config.max_autotune_gemm_backends = "ATEN,TRITON"
    return True


__all__ = ["configure_flex_attention_inductor"]
