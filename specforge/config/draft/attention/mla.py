"""Latent attention projection geometry and rotary convention."""

from dataclasses import dataclass
from typing import ClassVar

from .._validation import boolean, finite_number, positive_int
from .base import AttentionSpec


@dataclass(frozen=True, kw_only=True)
class MLASpec(AttentionSpec):
    """Latent projections and rotary subspace; no GQA head/cache assumptions."""

    type: ClassVar[str] = "mla"
    capabilities: ClassVar[frozenset[str]] = frozenset({"target_context_kv"})

    num_attention_heads: int
    q_lora_rank: int | None
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rope_interleave: bool
    attention_bias: bool
    is_causal: bool
    sliding_window: int | None
    attention_dropout: float

    def __post_init__(self):
        super().__post_init__()
        for name in (
            "num_attention_heads",
            "kv_lora_rank",
            "qk_rope_head_dim",
            "v_head_dim",
        ):
            positive_int(name, getattr(self, name))
        if self.q_lora_rank is not None:
            positive_int("q_lora_rank", self.q_lora_rank)
        if type(self.qk_nope_head_dim) is not int or self.qk_nope_head_dim < 0:
            raise ValueError("qk_nope_head_dim must be a nonnegative integer")
        if self.qk_rope_head_dim % 2:
            raise ValueError("qk_rope_head_dim must be even")
        for name in ("rope_interleave", "attention_bias", "is_causal"):
            boolean(name, getattr(self, name))
        if self.sliding_window is not None:
            positive_int("sliding_window", self.sliding_window)
        finite_number("attention_dropout", self.attention_dropout)
        if not 0 <= self.attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0, 1)")

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def rotary_input_dim(self) -> int:
        return self.qk_rope_head_dim
