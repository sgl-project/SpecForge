"""Kimi Delta Attention: a gated delta-rule recurrence instead of a KV lookup."""

from dataclasses import dataclass
from typing import ClassVar

from .._validation import boolean, finite_number, positive_int
from .base import AttentionSpec


@dataclass(frozen=True, kw_only=True)
class KDASpec(AttentionSpec):
    """Per-head recurrent state over the visible target context, then each block.

    Field names follow the Kimi ``linear_attn_config``. The decay gate is low
    rank; ``full_rank_output_gate`` replaces the low-rank output gate with one
    projection. ``gate_lower_bound`` selects Kimi's bounded ("safe") decay gate.
    KDA has no rotary subspace, so ``rope_parameters`` must be None.
    """

    type: ClassVar[str] = "kda"

    def capabilities(self) -> frozenset[str]:
        return frozenset({"target_context_state"})

    num_heads: int
    head_dim: int
    short_conv_kernel_size: int
    gate_lower_bound: float | None
    full_rank_output_gate: bool

    def __post_init__(self):
        super().__post_init__()
        for name in ("num_heads", "head_dim", "short_conv_kernel_size"):
            positive_int(name, getattr(self, name))
        if self.gate_lower_bound is not None:
            finite_number("gate_lower_bound", self.gate_lower_bound)
            if self.gate_lower_bound >= 0:
                raise ValueError("gate_lower_bound must be negative or null")
        boolean("full_rank_output_gate", self.full_rank_output_gate)

    @property
    def projection_size(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def rotary_input_dim(self) -> None:
        return None
