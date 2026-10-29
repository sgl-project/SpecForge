"""P-EAGLE parameters."""

from dataclasses import dataclass
from typing import Self

from transformers import LlamaConfig

from .base import boolean
from .eagle3 import EagleConfig


@dataclass(frozen=True, kw_only=True)
class PEagleConfig(EagleConfig):
    """P-EAGLE feature settings and normalization ordering around the residual."""

    algorithm = "peagle"

    norm_before_residual: bool

    @classmethod
    def from_model_config(cls, config: LlamaConfig) -> Self:
        return cls(
            **cls._common_fields(config),
            norm_before_residual=boolean(
                getattr(config, "norm_before_residual", False), "norm_before_residual"
            ),
        )
