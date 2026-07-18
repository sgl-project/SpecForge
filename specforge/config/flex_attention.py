"""Validation shared by canonical and production FlexAttention configs."""

from __future__ import annotations

from typing import Any, Optional

FLEX_KERNEL_OPTION_KEYS = frozenset(
    {
        "BACKEND",
        "BLOCKS_ARE_CONTIGUOUS",
        "BLOCK_M",
        "BLOCK_M1",
        "BLOCK_M2",
        "BLOCK_N",
        "BLOCK_N1",
        "BLOCK_N2",
        "FORCE_USE_FLEX_ATTENTION",
        "PRESCALE_QK",
        "ROWS_GUARANTEED_SAFE",
        "USE_TMA",
        "WRITE_DQ",
        "kpack",
        "matrix_instr_nonkdim",
        "num_stages",
        "num_warps",
        "waves_per_eu",
    }
)


def validate_flex_kernel_options(
    value: Optional[dict[str, Any]], *, field_name: str
) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    unknown = sorted(set(value) - FLEX_KERNEL_OPTION_KEYS)
    if unknown:
        raise ValueError(f"{field_name} contains unsupported keys: {unknown}")
    if "num_stages" in value:
        num_stages = value["num_stages"]
        if (
            isinstance(num_stages, bool)
            or not isinstance(num_stages, int)
            or num_stages < 1
        ):
            raise ValueError(f"{field_name}.num_stages must be a positive integer")
    return value


__all__ = ["FLEX_KERNEL_OPTION_KEYS", "validate_flex_kernel_options"]
