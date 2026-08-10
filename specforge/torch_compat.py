"""PyTorch compatibility shims used by optional fast paths."""

from __future__ import annotations

import importlib

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
