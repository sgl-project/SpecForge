"""domino parameters and algorithm-specific validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import Qwen3Config

from .base import integer as _integer
from .dflash import DFlashConfig, _projector


@dataclass(frozen=True, kw_only=True)
class DominoConfig(DFlashConfig):
    """Domino retains DFlash block_size semantics, including the anchor.

    emb_dim: Token embedding projection width for the prefix recurrence.
    gru_hidden_dim: Prefix GRU state width.
    """

    algorithm = "domino"

    emb_dim: int
    gru_hidden_dim: int

    @classmethod
    def _variant_fields(
        cls, config: Qwen3Config, method: dict[str, Any], mode: str, block_size: int
    ) -> dict[str, Any]:
        return {
            "projector_type": _projector(method, "domino"),
            "emb_dim": _integer(method.get("emb_dim"), "emb_dim"),
            "gru_hidden_dim": _integer(method.get("gru_hidden_dim"), "gru_hidden_dim"),
        }
