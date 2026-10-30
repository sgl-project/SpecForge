"""Immutable component specifications. No model implementation is imported here."""

from __future__ import annotations

from dataclasses import dataclass

from .base import boolean, integer


@dataclass(frozen=True, kw_only=True)
class AttentionSpec:
    type: str
    num_attention_heads: int
    is_causal: bool
    sliding_window: int | None = None

    def __post_init__(self):
        integer(self.num_attention_heads, "num_attention_heads")
        boolean(self.is_causal, "is_causal")
        if self.sliding_window is not None:
            integer(self.sliding_window, "sliding_window")

    @property
    def layer_type(self):
        return (
            "sliding_attention" if self.sliding_window is not None else "full_attention"
        )


@dataclass(frozen=True, kw_only=True)
class GQASpec(AttentionSpec):
    type: str = "gqa"
    num_key_value_heads: int
    head_dim: int

    def __post_init__(self):
        super().__post_init__()
        integer(self.num_key_value_heads, "num_key_value_heads")
        integer(self.head_dim, "head_dim")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_key_value_heads must divide num_attention_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even")
        if self.type == "mha" and self.num_key_value_heads != self.num_attention_heads:
            raise ValueError("mha requires num_key_value_heads == num_attention_heads")

    @property
    def rotary_dim(self):
        return self.head_dim


@dataclass(frozen=True, kw_only=True)
class MLASpec(AttentionSpec):
    type: str = "mla"
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    q_lora_rank: int | None = None
    rope_interleave: bool = True

    def __post_init__(self):
        super().__post_init__()
        for name in ("kv_lora_rank", "qk_rope_head_dim", "v_head_dim"):
            integer(getattr(self, name), name)
        integer(self.qk_nope_head_dim, "qk_nope_head_dim", 0)
        if self.q_lora_rank is not None:
            integer(self.q_lora_rank, "q_lora_rank")
        if self.qk_rope_head_dim % 2:
            raise ValueError("qk_rope_head_dim must be even")
        boolean(self.rope_interleave, "rope_interleave")

    @property
    def rotary_dim(self):
        return self.qk_rope_head_dim


@dataclass(frozen=True, kw_only=True)
class FFNSpec:
    type: str = "dense"
    intermediate_size: int

    def __post_init__(self):
        integer(self.intermediate_size, "intermediate_size")


@dataclass(frozen=True, kw_only=True)
class ResidualSpec:
    type: str = "standard"


@dataclass(frozen=True, kw_only=True)
class ConvSpec:
    type: str = "grouped"
    kernel_size: int
    group_size: int

    def __post_init__(self):
        integer(self.kernel_size, "conv_kernel_size")
        integer(self.group_size, "conv_group_size")


@dataclass(frozen=True, kw_only=True)
class LayerSpec:
    attention: AttentionSpec
    ffn: FFNSpec
    residual: ResidualSpec
    conv: ConvSpec | None = None
