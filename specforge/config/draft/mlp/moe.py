"""Mixture-of-experts feed-forward structure, serialized as mlp_configs."""

from dataclasses import dataclass
from typing import ClassVar

from .._validation import boolean, nonempty_string, positive_int, positive_number
from .base import FFNSpec
from .dense import SUPPORTED_ACTIVATIONS

# Router affinity functions. softmax is Qwen3.5-MoE; sigmoid is DeepSeek-V3/V4.
ROUTER_SCORINGS = frozenset({"softmax", "sigmoid"})


@dataclass(frozen=True, kw_only=True)
class MoERouterSpec:
    """Top-k routing on `scoring` affinities; `expert_bias` shifts the choice only.

    Weights are the unshifted affinities, renormalized if `normalize_top_k`, times `route_scale`.
    """

    scoring: str
    normalize_top_k: bool
    route_scale: float
    expert_bias: bool

    def __post_init__(self):
        nonempty_string("scoring", self.scoring)
        if self.scoring not in ROUTER_SCORINGS:
            raise ValueError(
                f"unsupported router scoring {self.scoring!r}; "
                f"supported: {sorted(ROUTER_SCORINGS)}"
            )
        boolean("normalize_top_k", self.normalize_top_k)
        positive_number("route_scale", self.route_scale)
        boolean("expert_bias", self.expert_bias)


@dataclass(frozen=True, kw_only=True)
class MoEFFNSpec(FFNSpec):
    """Routed gated experts of per-expert width `intermediate_size`, plus an optional
    shared expert (`null` width = none) with a sigmoid gate when `shared_expert_gate`.
    """

    type: ClassVar[str] = "moe"

    num_experts: int
    num_experts_per_tok: int
    intermediate_size: int
    hidden_act: str
    shared_expert_intermediate_size: int | None
    shared_expert_gate: bool
    router: MoERouterSpec

    def __post_init__(self):
        positive_int("num_experts", self.num_experts)
        positive_int("num_experts_per_tok", self.num_experts_per_tok)
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError(
                f"num_experts_per_tok {self.num_experts_per_tok} exceeds "
                f"num_experts {self.num_experts}"
            )
        positive_int("intermediate_size", self.intermediate_size)
        nonempty_string("hidden_act", self.hidden_act)
        if self.hidden_act not in SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"unsupported moe hidden_act {self.hidden_act!r}; "
                f"supported: {sorted(SUPPORTED_ACTIVATIONS)}"
            )
        if self.shared_expert_intermediate_size is not None:
            positive_int(
                "shared_expert_intermediate_size",
                self.shared_expert_intermediate_size,
            )
        boolean("shared_expert_gate", self.shared_expert_gate)
        if self.shared_expert_gate and self.shared_expert_intermediate_size is None:
            raise ValueError("shared_expert_gate requires a shared expert")
        if not isinstance(self.router, MoERouterSpec):
            raise TypeError("router must be MoERouterSpec")
