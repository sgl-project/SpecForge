"""Mixture-of-experts feed-forward structure, serialized as mlp_configs."""

from dataclasses import dataclass
from typing import ClassVar

from .._validation import boolean, nonempty_string, positive_int, positive_number
from .base import FFNSpec
from .dense import SUPPORTED_ACTIVATIONS

# Router affinity functions. softmax is Qwen3.5-MoE; sigmoid is DeepSeek-V3/V4.
ROUTER_SCORINGS = frozenset({"softmax", "sigmoid"})
# Expert activations: the dense set plus relu2 for Nemotron's non-gated MLP.
EXPERT_ACTIVATIONS = SUPPORTED_ACTIVATIONS | {"relu2"}


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
class MoEExpertSpec:
    """One expert FFN of width `intermediate_size`; `gated` is act(gate_proj)·up_proj as in
    Qwen and DeepSeek, else act(up_proj) as in Nemotron's relu2 MLP. The routed experts
    and the shared expert use it alike (DeepSeek merges n shared experts into n×I).
    """

    intermediate_size: int
    hidden_act: str
    gated: bool

    def __post_init__(self):
        positive_int("intermediate_size", self.intermediate_size)
        nonempty_string("hidden_act", self.hidden_act)
        if self.hidden_act not in EXPERT_ACTIVATIONS:
            raise ValueError(
                f"unsupported expert hidden_act {self.hidden_act!r}; "
                f"supported: {sorted(EXPERT_ACTIVATIONS)}"
            )
        boolean("gated", self.gated)


@dataclass(frozen=True, kw_only=True)
class MoEFFNSpec(FFNSpec):
    """`num_experts_per_tok` of `num_experts` routed `experts` per token, plus an optional
    `shared_expert` whose output a sigmoid gate scales when `shared_expert_gate` (Qwen).
    """

    type: ClassVar[str] = "moe"

    num_experts: int
    num_experts_per_tok: int
    experts: MoEExpertSpec
    shared_expert: MoEExpertSpec | None
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
        if not isinstance(self.experts, MoEExpertSpec):
            raise TypeError("experts must be MoEExpertSpec")
        if not self.experts.gated:
            raise ValueError("routed experts support gated FFNs only")
        if self.shared_expert is not None and not isinstance(
            self.shared_expert, MoEExpertSpec
        ):
            raise TypeError("shared_expert must be MoEExpertSpec or None")
        boolean("shared_expert_gate", self.shared_expert_gate)
        if self.shared_expert_gate and self.shared_expert is None:
            raise ValueError("shared_expert_gate requires a shared expert")
        if not isinstance(self.router, MoERouterSpec):
            raise TypeError("router must be MoERouterSpec")
