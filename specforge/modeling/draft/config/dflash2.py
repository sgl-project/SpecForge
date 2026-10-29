"""dflash2 parameters and algorithm-specific validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from transformers import Qwen3Config

from .base import integer as _integer
from .dflash import DFlashConfig


@dataclass(frozen=True, kw_only=True)
class DFlash2Config(DFlashConfig):
    """DFlash2 retains DFlash block_size semantics, including the anchor.

    conv_kernel_size: Width of the within-block causal convolution.
    conv_group_size: Channels per convolution group; must divide hidden_size.
    selector_rank: Low-rank candidate selector width.
    selector_top_k: Number of vocabulary candidates selected per position.
    output_multiplier: Scale applied to output logits.
    final_logit_softcapping: Optional positive output logit cap.
    """

    algorithm = "dflash2"

    conv_kernel_size: int
    conv_group_size: int
    selector_rank: int
    selector_top_k: int
    output_multiplier: float
    final_logit_softcapping: float | None

    @classmethod
    def _variant_fields(
        cls, config: Qwen3Config, method: dict[str, Any], mode: str, block_size: int
    ) -> dict[str, Any]:
        values = super()._variant_fields(config, method, mode, block_size)
        for name in (
            "conv_kernel_size",
            "conv_group_size",
            "selector_rank",
            "selector_top_k",
        ):
            values[name] = _integer(method.get(name), f"dflash_config.{name}")
        if values["conv_kernel_size"] > block_size:
            raise ValueError("conv_kernel_size must not exceed block_size")
        if config.hidden_size % values["conv_group_size"]:
            raise ValueError("conv_group_size must divide hidden_size")
        if values["selector_top_k"] > config.vocab_size:
            raise ValueError("selector_top_k must be in [1, vocab_size]")
        multiplier = float(method.get("output_multiplier", 1.0))
        softcap = method.get("final_logit_softcapping")
        if not math.isfinite(multiplier):
            raise ValueError("output_multiplier must be finite")
        if softcap is not None:
            softcap = float(softcap)
            if not math.isfinite(softcap) or softcap <= 0:
                raise ValueError("final_logit_softcapping must be finite and > 0")
        return {
            **values,
            "output_multiplier": multiplier,
            "final_logit_softcapping": softcap,
        }
