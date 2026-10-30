from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from transformers import Qwen3Config

from .base import integer as _integer
from .dflash import DFlashConfig


@dataclass(frozen=True, kw_only=True)
class DFlash2Config(DFlashConfig):
    requires_conv = True
    selector_rank: int
    selector_top_k: int
    output_multiplier: float
    final_logit_softcapping: float | None

    @classmethod
    def _variant_fields(
        cls, config: Qwen3Config, method: dict[str, Any], block_size: int
    ) -> dict[str, Any]:
        values = super()._variant_fields(config, method, block_size)
        for name in (
            "selector_rank",
            "selector_top_k",
        ):
            values[name] = _integer(method.get(name), f"dflash_config.{name}")
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
