"""Parse user-supplied draft settings independently of model allocation.

The configuration registry owns the HF adapter and typed architecture options.
This loader does not infer the draft architecture from the target or construct
model parameters.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from transformers import PretrainedConfig

from .base import integer
from .model import DraftModelConfig, inline_values
from .registry import DraftConfigRegistration, resolve_draft_config_type


class AutoDraftConfig:
    """Parse embedded settings; checkpoint metadata is handled separately.

    Training uses from_inline. from_dict also decodes legacy metadata for
    compatibility checks, but never reads a checkpoint or HF repository.
    """

    @classmethod
    def from_inline(cls, payload: Mapping[str, Any]) -> DraftModelConfig:
        """Resolve the YAML draft section without parsing it through HF."""
        if not isinstance(payload, Mapping):
            raise ValueError("draft config must be an object")
        registration = cls._registration(payload.get("architectures"))
        values = inline_values(payload, model_type=registration.config_class.model_type)
        integer(values.draft_vocab_size, "draft_vocab_size")
        if registration.options_class is not None:
            options = registration.options_class.from_model_config(values)
            values = options.to_model_config(values)
        return DraftModelConfig(values.to_dict())

    @classmethod
    def resolve(cls, config, *, architecture: str | None = None) -> DraftModelConfig:
        """Accept a resolved config, or adapt an existing HF checkpoint config."""
        if isinstance(config, DraftModelConfig):
            return config
        if isinstance(config, Mapping):
            return cls.from_inline(config)
        payload = config.to_dict()
        # An explicitly selected model class also accepts legacy base-model
        # metadata (e.g. LlamaForCausalLM). Never rewrite resolved inline settings.
        if architecture is not None:
            payload["architectures"] = [architecture]
        payload["_attn_implementation"] = getattr(config, "_attn_implementation", None)
        return cls.from_inline(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PretrainedConfig:
        """Read legacy HF checkpoint metadata; training uses from_inline."""
        if not isinstance(payload, Mapping):
            raise ValueError("draft config must be a JSON object")
        payload = copy.deepcopy(dict(payload))
        registration = cls._registration(payload.get("architectures"))
        payload["architectures"] = list(payload["architectures"])
        payload["tie_word_embeddings"] = False
        payload.setdefault("use_cache", True)
        config = registration.config_class.from_dict(payload)
        if payload.get("draft_vocab_size") is None:
            config.draft_vocab_size = config.vocab_size
        if type(config.draft_vocab_size) is not int or config.draft_vocab_size <= 0:
            raise ValueError("draft config draft_vocab_size must be a positive integer")
        return config

    @classmethod
    def validate_training_config(
        cls, config: DraftModelConfig | PretrainedConfig
    ) -> None:
        """Check training-only requirements on a resolved config."""
        registration = cls._registration(config.architectures)
        if registration.options_class is not None:
            registration.options_class.validate_training_config(config)

    @staticmethod
    def _registration(architectures) -> DraftConfigRegistration:
        if (
            not isinstance(architectures, (list, tuple))
            or len(architectures) != 1
            or not isinstance(architectures[0], str)
        ):
            raise ValueError(
                "draft config must name exactly one registered architecture"
            )
        return resolve_draft_config_type(architectures[0])
