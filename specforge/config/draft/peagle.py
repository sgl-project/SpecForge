from dataclasses import dataclass
from typing import Self

from transformers import LlamaConfig

from .base import boolean
from .eagle3 import EagleConfig


@dataclass(frozen=True, kw_only=True)
class PEagleConfig(EagleConfig):
    attention_contract = "peagle"
    norm_before_residual: bool

    @classmethod
    def validate_training_config(cls, config) -> None:
        super().validate_training_config(config)
        mask = getattr(config, "mask_token_id", None)
        if type(mask) is not int or not 0 <= mask < config.vocab_size:
            raise ValueError("inline mask_token_id must be within vocab_size")

    @classmethod
    def from_model_config(cls, config: LlamaConfig) -> Self:
        return cls(
            **cls._common_fields(config),
            norm_before_residual=boolean(
                getattr(config, "norm_before_residual", False), "norm_before_residual"
            ),
        )
