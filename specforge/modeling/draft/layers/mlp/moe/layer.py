"""MoEFFN composing the router and experts, and its registration."""

import torch
from torch import Tensor, nn

from specforge.config.draft._validation import positive_int
from specforge.config.draft.mlp import MoEFFNSpec

from ...inputs import ExpertRouting, MoEOutput
from ...registry import register_component_builder
from .experts import RoutedExperts, SharedExpert
from .router import TopKRouter


class MoEFFN(nn.Module):
    """Top-k routed experts plus the optional shared expert, built from one MoEFFNSpec."""

    def __init__(
        self, config: MoEFFNSpec, *, hidden_size: int, device=None, dtype=None
    ):
        super().__init__()
        if type(config) is not MoEFFNSpec:
            raise TypeError("MoEFFN requires MoEFFNSpec")
        positive_int("hidden_size", hidden_size)
        self.config = config
        # A runtime choice, not state: configure_kernels resolves it once at startup.
        self.experts_kernel = "grouped_mm"
        self.gate = TopKRouter(
            config.router,
            num_experts=config.num_experts,
            hidden_size=hidden_size,
            top_k=config.num_experts_per_tok,
            device=device,
            dtype=dtype,
        )
        self.experts = RoutedExperts(
            config.experts,
            num_experts=config.num_experts,
            hidden_size=hidden_size,
            device=device,
            dtype=dtype,
        )
        self.shared_expert = (
            SharedExpert(
                config.shared_expert,
                hidden_size=hidden_size,
                device=device,
                dtype=dtype,
            )
            if config.shared_expert is not None
            else None
        )
        self.shared_expert_gate = (
            nn.Linear(hidden_size, 1, bias=False, device=device, dtype=dtype)
            if config.shared_expert_gate
            else None
        )

    def forward(self, hidden: Tensor) -> MoEOutput:
        """The stream update in `hidden`'s shape, typed together with its routing."""
        x = hidden.reshape(-1, hidden.shape[-1])
        scores, indices, weights = self.gate(x)
        out = self.experts(x, indices, weights, kernel=self.experts_kernel)
        if self.shared_expert is not None:
            shared = self.shared_expert(x)
            if self.shared_expert_gate is not None:
                shared = torch.sigmoid(self.shared_expert_gate(x)) * shared
            out = out + shared
        # The routed sum is fp32; cast back once, after adding the shared expert.
        update = out.to(hidden.dtype).view(hidden.shape)
        return MoEOutput(update=update, routing=ExpertRouting(indices, scores))


register_component_builder("mlp", MoEFFNSpec, MoEFFN)
