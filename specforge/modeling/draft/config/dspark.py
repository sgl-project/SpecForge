"""dspark parameters and algorithm-specific validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import Qwen3Config

from .base import boolean as _boolean
from .base import integer as _integer
from .dflash import DFlashConfig, _projector


@dataclass(frozen=True, kw_only=True)
class DSparkConfig(DFlashConfig):
    """DSpark block_size counts proposed tokens (verification adds the anchor).

    markov_rank: Auxiliary Markov head rank; zero disables the head.
    markov_head_type: vanilla, gated, or rnn parameterization.
    enable_confidence_head: Allocate per-position acceptance prediction weights.
    confidence_head_with_markov: Feed Markov features to the confidence head.
    """

    algorithm = "dspark"

    markov_rank: int
    markov_head_type: str
    enable_confidence_head: bool
    confidence_head_with_markov: bool

    @classmethod
    def _variant_fields(
        cls, config: Qwen3Config, method: dict[str, Any], mode: str, block_size: int
    ) -> dict[str, Any]:
        if mode == "gqa" and config.num_key_value_heads == config.num_attention_heads:
            raise ValueError(
                "DSpark defaults to GQA; set attention_mode='mha' for equal query/KV heads"
            )
        # Legacy configs used null to disable the optional Markov head.
        rank = method.get("markov_rank")
        rank = _integer(0 if rank is None else rank, "markov_rank", 0)
        head_type = str(method.get("markov_head_type", "vanilla")).lower()
        if rank and head_type not in ("vanilla", "gated", "rnn"):
            raise ValueError(f"Unsupported markov_head_type: {head_type!r}")
        enabled = method.get("enable_confidence_head")
        if "enable_confidence_head" not in method:
            enabled = float(method.get("confidence_head_alpha", 0.0) or 0.0) > 0
        with_markov = _boolean(
            method.get("confidence_head_with_markov", False),
            "confidence_head_with_markov",
        )
        if with_markov and rank == 0:
            raise ValueError(
                "confidence_head_with_markov=True requires markov_rank > 0"
            )
        return {
            "projector_type": _projector(method, "dspark"),
            "markov_rank": rank,
            "markov_head_type": head_type,
            "enable_confidence_head": _boolean(enabled, "enable_confidence_head"),
            "confidence_head_with_markov": with_markov,
        }
