"""Attention structure; runtime backends and model dimensions live elsewhere."""

from dataclasses import dataclass
from typing import ClassVar

from .._validation import boolean, finite_number, positive_int
from ..norm import NormSpec
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
    qk_norm_position: str
    qk_norm_scope: str
    qk_norm_config: NormSpec | None

    def __post_init__(self):
        super().__post_init__()
        if self.qk_norm_position not in ("before_rope", "after_rope"):
            raise ValueError("qk_norm_position must be before_rope or after_rope")
        if self.qk_norm_scope not in ("head", "projection"):
            raise ValueError("qk_norm_scope must be head or projection")
        if self.qk_norm_config is not None and not isinstance(
            self.qk_norm_config, NormSpec
        ):
            raise ValueError("qk_norm_config must be a norm node or None")
        if not self.qk_norm and (
            self.qk_norm_config is not None
            or self.qk_norm_position != "before_rope"
            or self.qk_norm_scope != "head"
        ):
            raise ValueError("qk_norm settings require qk_norm=True")
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
