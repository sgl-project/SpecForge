"""PyTorch compatibility shims used by optional fast paths."""

from __future__ import annotations

import importlib
import os

import sympy
import torch
from packaging.version import InvalidVersion, Version


def patch_inductor_cutedsl_lowerings() -> bool:
    """Backfill CuteDSL lowering needed by Torch 2.11 FLASH FlexAttention."""
    try:
        torch_version = Version(torch.__version__.split("+", 1)[0])
    except InvalidVersion:
        return False
    if torch_version.major != 2 or torch_version.minor != 11:
        return False

    try:
        module = importlib.import_module(
            "torch._inductor.codegen.cutedsl.cutedsl_op_overrides"
        )
    except ImportError:
        return False
    from torch._inductor.utils import get_bounds_index_expr
    from torch._inductor.virtualized import V

    overrides = module.CuteDSLOpOverrides
    if getattr(overrides, "_specforge_cutedsl_patch", False):
        return True

    def _minimum(a, b):
        return overrides.where(overrides.lt(a, b), a, b)

    def _maximum(a, b):
        return overrides.where(overrides.gt(a, b), a, b)

    def _index_expr(expr: sympy.Expr, dtype: torch.dtype):
        if isinstance(expr, (int, sympy.Integer)):
            return overrides.constant(int(expr), dtype)

        idx_str = V.kernel.kexpr(V.kernel.rename_indexing(expr))
        result = V.kernel.cse.generate(
            V.kernel.body,
            idx_str,
            bounds=get_bounds_index_expr(expr),
            dtype=dtype,
        )
        result.is_scalar_expr = True
        result.index_expr = V.graph.sizevars.simplify(expr)
        return result

    overrides.minimum = staticmethod(_minimum)
    overrides.maximum = staticmethod(_maximum)
    overrides.index_expr = staticmethod(_index_expr)
    overrides._specforge_cutedsl_patch = True
    return True


__all__ = ["patch_inductor_cutedsl_lowerings"]


_INDUCTOR_GEMM_BACKENDS = "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"


def configure_flex_attention_inductor(attention_backend: str) -> bool:
    """Pin TorchInductor's GEMM autotune candidates to ``ATEN,TRITON``.

    Colocated capture feeds raw variable-length sequences into the compiled
    FlexAttention draft forward. The colocated validation run hit
    ``NoValidChoicesError`` (the autotuner found no valid candidate) on some of
    those shapes, and did not once ``TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS``
    was set to ``ATEN,TRITON``. torch 2.11's default already lists ``ATEN``, so
    the effective change is dropping ``CPP`` from the candidate set; the exact
    mechanism has not been isolated. An explicit operator setting of the
    variable stays authoritative. Returns ``True`` when this call installed the
    default.
    """
    if attention_backend != "flex_attention" or _INDUCTOR_GEMM_BACKENDS in os.environ:
        return False
    os.environ[_INDUCTOR_GEMM_BACKENDS] = "ATEN,TRITON"
    # torch._inductor.config reads the variable once at import; it may already
    # be imported (Transformers, Accelerate, an in-process SGLang engine), so
    # update the live value as well.
    from torch._inductor import config as inductor_config

    inductor_config.max_autotune_gemm_backends = "ATEN,TRITON"
    return True
