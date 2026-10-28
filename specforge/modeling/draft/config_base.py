"""Shared contract and scalar validation for resolved draft architecture options."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import ClassVar, Self

from transformers import PretrainedConfig


def integer(value: object, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean, got {value!r}")
    return value


def method_config(config: PretrainedConfig, name: str) -> dict:
    value = getattr(config, name, None)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def validate_dimensions(config: PretrainedConfig) -> None:
    for name in (
        "hidden_size",
        "intermediate_size",
        "vocab_size",
        "num_hidden_layers",
        "num_attention_heads",
    ):
        integer(getattr(config, name), name)
    integer(getattr(config, "draft_vocab_size", config.vocab_size), "draft_vocab_size")


def validate_gqa(config: PretrainedConfig) -> int:
    heads = integer(config.num_attention_heads, "num_attention_heads")
    kv_heads = integer(config.num_key_value_heads, "num_key_value_heads")
    if heads % kv_heads:
        raise ValueError("num_key_value_heads must divide num_attention_heads")
    return integer(getattr(config, "head_dim", config.hidden_size // heads), "head_dim")


@dataclass(frozen=True, kw_only=True)
class DraftConfig(ABC):
    """Immutable model-specific options; HF config remains the checkpoint format.

    Parsing does not allocate tensors or mutate its input. Subclasses own their
    legacy JSON layout and retain fields they do not interpret.
    """

    _nested_fields: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    @abstractmethod
    def from_model_config(cls, config: PretrainedConfig) -> Self:
        raise NotImplementedError

    def to_model_config(self, config: PretrainedConfig) -> PretrainedConfig:
        result = copy.deepcopy(config)
        for field in fields(self):
            if field.name not in self._nested_fields:
                setattr(result, field.name, getattr(self, field.name))
        result.tie_word_embeddings = False
        result.draft_vocab_size = getattr(config, "draft_vocab_size", config.vocab_size)
        return result
