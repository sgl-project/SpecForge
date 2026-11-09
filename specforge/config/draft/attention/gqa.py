"""Attention structure; runtime backends and model dimensions live elsewhere."""

from dataclasses import dataclass
from typing import ClassVar

from .._validation import boolean, finite_number, positive_int
from .base import AttentionSpec


@dataclass(frozen=True, kw_only=True)
class GQASpec(AttentionSpec):
    type: ClassVar[str] = "gqa"

    def capabilities(self) -> frozenset[str]:
        flags = {"target_context_kv", "variable_input_width"}
        if self.is_causal:
            flags.add("causal")
        return frozenset(flags)

    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    qk_norm: bool
    attention_bias: bool
    output_gate: bool
    is_causal: bool
    sliding_window: int | None
    attention_dropout: float

    def __post_init__(self):
        super().__post_init__()
        for name in ("num_attention_heads", "num_key_value_heads", "head_dim"):
            positive_int(name, getattr(self, name))
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        for name in ("qk_norm", "attention_bias", "output_gate", "is_causal"):
            boolean(name, getattr(self, name))
        if self.sliding_window is not None:
            positive_int("sliding_window", self.sliding_window)
        finite_number("attention_dropout", self.attention_dropout)
        if not 0 <= self.attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0, 1)")

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def rotary_input_dim(self) -> int:
        return self.head_dim


@dataclass(frozen=True, kw_only=True)
class MHASpec(GQASpec):
    type: ClassVar[str] = "mha"

    def __post_init__(self):
        super().__post_init__()
        if self.num_attention_heads != self.num_key_value_heads:
            raise ValueError(
                "mha requires equal num_attention_heads and num_key_value_heads"
            )
