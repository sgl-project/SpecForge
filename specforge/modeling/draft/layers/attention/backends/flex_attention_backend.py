"""FlexAttention backend and kernel options shared by DFlash training components."""

from __future__ import annotations

import os
from typing import Optional

import torch

from specforge.torch_compat import patch_inductor_cutedsl_lowerings

_VALID_BACKENDS = {"AUTO", "TRITON", "FLASH", "TRITON_DECODE"}
_BACKEND_ENV = "SPECFORGE_FLEX_ATTENTION_BACKEND"


def flex_attention_backend() -> Optional[str]:
    backend = os.environ.get(_BACKEND_ENV, "").upper()
    if not backend:
        return None
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"{_BACKEND_ENV} must be one of {sorted(_VALID_BACKENDS)}, "
            f"got {backend!r}"
        )
    if backend == "FLASH":
        patch_inductor_cutedsl_lowerings()
    return backend


def flex_kernel_options(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kernel_options: dict
) -> tuple[dict, torch.Tensor]:
    """Fill hardware defaults for head_dim > 256; the caller's options win."""
    backend = flex_attention_backend()
    if backend is not None:
        kernel_options["BACKEND"] = backend
    major = torch.cuda.get_device_capability(q.device)[0] if q.is_cuda else 0
    if q.shape[-1] > 256 and major >= 9:
        # torch 2.13 picks 16x16 backward tiles past head_dim 256; 32x32 runs 3x faster.
        for key in ("BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2"):
            kernel_options.setdefault(key, 32)
        kernel_options.setdefault("num_warps", 8)
        kernel_options.setdefault("num_stages", 3)
    if q.shape[-1] > 256 and major >= 10:
        # SM100 faults on the default forward tiles and crawls when key and value alias.
        kernel_options.setdefault("BLOCK_M", 32)
        kernel_options.setdefault("BLOCK_N", 32)
        if v is k:
            v = k.clone()
    return kernel_options, v


__all__ = ["flex_attention_backend", "flex_kernel_options"]
