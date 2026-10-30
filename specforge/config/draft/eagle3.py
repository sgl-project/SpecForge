"""EAGLE3 and P-EAGLE architecture options in their existing HF JSON layout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from huggingface_hub.dataclasses import strict
from transformers import LlamaConfig

from .base import (
    DraftConfig,
    boolean,
    integer,
    method_config,
    validate_dimensions,
    validate_gqa,
)


@strict
class EaglePretrainedConfig(LlamaConfig):
    """Llama JSON compatibility with EAGLE's independent projection width."""

    def validate_architecture(self):
        # EAGLE projects hidden_size to num_heads * head_dim and back. These
        # widths need not match; upstream Llama's divisibility check is stronger
        # than the draft requires. Keep the actual GQA dimension constraints.
        validate_gqa(self)


@dataclass(frozen=True, kw_only=True)
class EagleConfig(DraftConfig):
    _nested_fields = frozenset({"aux_hidden_state_layer_ids"})
    target_hidden_size: int
    aux_hidden_state_layer_ids: tuple[int, ...] | None

    @classmethod
    def validate_training_config(cls, config) -> None:
        layers = method_config(config, "eagle_config").get(
            "eagle_aux_hidden_state_layer_ids"
        )
        if layers is None or len(set(layers)) != 3:
            raise ValueError(
                "inline eagle_config.eagle_aux_hidden_state_layer_ids must contain "
                "three distinct layer indices"
            )

    @classmethod
    def _common_fields(cls, config: LlamaConfig) -> dict:
        validate_dimensions(config)
        if validate_gqa(config) % 2:
            raise ValueError("head_dim must be even for EAGLE rotary embeddings")
        layers = method_config(config, "eagle_config").get(
            "eagle_aux_hidden_state_layer_ids"
        )
        if layers is not None:
            if not isinstance(layers, (list, tuple)) or len(layers) != 3:
                raise ValueError(
                    "eagle_aux_hidden_state_layer_ids must contain three layer indices"
                )
            for layer in layers:
                integer(layer, "eagle_aux_hidden_state_layer_ids entry", 0)
            layers = tuple(layers)
        return {
            "target_hidden_size": integer(
                getattr(config, "target_hidden_size", config.hidden_size),
                "target_hidden_size",
            ),
            "aux_hidden_state_layer_ids": layers,
        }

    def to_model_config(self, config: LlamaConfig) -> LlamaConfig:
        result = super().to_model_config(config)
        # Capture layers retain the historical nested key, not a new top-level alias.
        method = dict(method_config(result, "eagle_config"))
        if self.aux_hidden_state_layer_ids is not None:
            method["eagle_aux_hidden_state_layer_ids"] = list(
                self.aux_hidden_state_layer_ids
            )
        else:
            method.pop("eagle_aux_hidden_state_layer_ids", None)
        result.eagle_config = method
        return result


@dataclass(frozen=True, kw_only=True)
class Eagle3Config(EagleConfig):
    fc_norm: bool
    norm_output: bool

    @classmethod
    def from_model_config(cls, config: LlamaConfig) -> Self:
        common = cls._common_fields(config)
        if config.num_hidden_layers != 1:
            raise ValueError("EAGLE3 requires num_hidden_layers=1")
        return cls(
            **common,
            fc_norm=boolean(getattr(config, "fc_norm", False), "fc_norm"),
            norm_output=boolean(getattr(config, "norm_output", False), "norm_output"),
        )
