"""Routers: the contract MoEFFN and balancing rely on, and the top-k implementation."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import initialization as init

from specforge.config.draft._validation import positive_int
from specforge.config.draft.mlp import MoERouterSpec


class Router(nn.Module):
    """Router contract: `forward`, and the fp32 `expert_bias` that shifts the top-k choice."""

    def __init__(
        self, config: MoERouterSpec, *, num_experts: int, top_k: int, device=None
    ):
        super().__init__()
        if type(config) is not MoERouterSpec:
            raise TypeError("Router requires MoERouterSpec")
        positive_int("num_experts", num_experts)
        positive_int("top_k", top_k)
        self.config = config
        self.num_experts = num_experts
        self.top_k = top_k
        # fp32 and persistent: a routing input that serving must apply too.
        self.register_buffer(
            "expert_bias",
            (
                torch.zeros(num_experts, dtype=torch.float32, device=device)
                if config.expert_bias
                else None
            ),
        )

    def _apply(self, fn, recurse=True):
        # Casts keep the bias in fp32 with its values; moves follow the parameters.
        old = self.expert_bias
        super()._apply(fn, recurse)
        new = self.expert_bias
        if old is not None and new.dtype != torch.float32:
            self.expert_bias = new.float() if old.is_meta else old.to(new.device)
        return self

    def reset_draft_parameters(self, *, initializer_range: float) -> None:
        if self.expert_bias is not None:
            init.zeros_(self.expert_bias)

    def _check_bias(self, bias: Tensor) -> None:
        if bias.dtype != torch.float32 or tuple(bias.shape) != (self.num_experts,):
            raise ValueError("expert_bias must be fp32 with one entry per expert")

    def validate_draft_state(self) -> None:
        if self.config.expert_bias != (self.expert_bias is not None):
            raise ValueError("expert_bias buffer disagrees with the router node")
        if self.expert_bias is not None:
            self._check_bias(self.expert_bias)

    def validate_export_state(self, state) -> None:
        expected = {name for name, _ in self.named_parameters()}
        if self.config.expert_bias:
            expected.add("expert_bias")
        if set(state) != expected:
            raise ValueError(f"router state {sorted(state)} must be {sorted(expected)}")
        if self.config.expert_bias:
            self._check_bias(state["expert_bias"])

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """`x` [T, H] -> fp32 scores [T, E], indices [T, K], fp32 weights [T, K]."""
        raise NotImplementedError


class TopKRouter(Router):
    """Linear gate `weight` [E, H] as Qwen's `gate`; biased top-k over `scoring` affinities."""

    def __init__(
        self,
        config: MoERouterSpec,
        *,
        num_experts: int,
        hidden_size: int,
        top_k: int,
        device=None,
        dtype=None,
    ):
        super().__init__(config, num_experts=num_experts, top_k=top_k, device=device)
        positive_int("hidden_size", hidden_size)
        self.weight = nn.Parameter(
            torch.empty(num_experts, hidden_size, device=device, dtype=dtype)
        )

    def reset_draft_parameters(self, *, initializer_range: float) -> None:
        super().reset_draft_parameters(initializer_range=initializer_range)
        init.normal_(self.weight, mean=0.0, std=initializer_range)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Logits stay in `x`'s dtype as in Qwen; scoring and the rest run in fp32."""
        logits = F.linear(x, self.weight).float()
        scores = (
            logits.softmax(dim=-1)
            if self.config.scoring == "softmax"
            else torch.sigmoid(logits)
        )
        # The bias shifts the choice only; the weights come from the unshifted scores.
        choice = scores if self.expert_bias is None else scores + self.expert_bias
        indices = choice.topk(self.top_k, dim=-1).indices
        weights = scores.gather(-1, indices)
        if self.config.normalize_top_k:
            # DeepSeek's 1e-20: a sigmoid top-k sum can be zero; a no-op for softmax.
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return scores, indices, weights * self.config.route_scale
