"""EAGLE3 and P-EAGLE architecture options in their existing HF JSON layout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from huggingface_hub.dataclasses import strict
from transformers import LlamaConfig

from specforge.modeling.draft.config.base import (
    DraftConfig,
    boolean,
    integer,
    method_config,
    validate_dimensions,
    validate_gqa,
)

from .components import validate_components
from .hf import DraftSerializationMixin


@strict
class EaglePretrainedConfig(DraftSerializationMixin, LlamaConfig):
    """Llama JSON compatibility with EAGLE's independent projection width."""

    def validate_architecture(self):
        # EAGLE projects hidden_size to num_heads * head_dim and back. These
        # widths need not match; upstream Llama's divisibility check is stronger
        # than the draft requires. Keep the actual GQA dimension constraints.
        validate_gqa(self)


@dataclass(frozen=True, kw_only=True)
class EagleConfig(DraftConfig):
    legacy_section = "eagle_config"
    non_spec_fields = frozenset({"aux_hidden_state_layer_ids"})
    top_level_fields = frozenset(
        {"target_hidden_size", "fc_norm", "norm_output", "norm_before_residual"}
    )
    spec_aliases = {
        "target_layer_ids": ("eagle_config", "eagle_aux_hidden_state_layer_ids")
    }

    _nested_fields = frozenset({"aux_hidden_state_layer_ids"})
    target_hidden_size: int
    aux_hidden_state_layer_ids: tuple[int, ...] | None

    @classmethod
    def _common_fields(cls, config: LlamaConfig) -> dict:
        validate_components(config, attention_types=("gqa", "mha"), causal_only=True)
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
    """EAGLE3 feature width/layers and optional fusion/output normalization."""

    algorithm = "eagle3"

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
