"""MoE FFN components: the top-k router, fused routed experts and the shared expert."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import initialization as init
from transformers.activations import ACT2FN

from specforge.config.draft._validation import positive_int
from specforge.config.draft.mlp import MoERouterSpec, MoESharedExpertSpec


class Router(nn.Module):
    """Linear gate scored by `config.scoring`; `expert_bias` shifts the top-k choice only."""

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
        super().__init__()
        if type(config) is not MoERouterSpec:
            raise TypeError("Router requires MoERouterSpec")
        for name, value in (
            ("num_experts", num_experts),
            ("hidden_size", hidden_size),
            ("top_k", top_k),
        ):
            positive_int(name, value)
        if top_k > num_experts:
            raise ValueError(f"top_k {top_k} exceeds num_experts {num_experts}")
        self.config = config
        self.num_experts = num_experts
        self.top_k = top_k
        self.weight = nn.Parameter(
            torch.empty(num_experts, hidden_size, device=device, dtype=dtype)
        )
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
        # Casts keep the bias in fp32 with its values; moves follow the weight.
        old = self.expert_bias
        super()._apply(fn, recurse)
        new = self.expert_bias
        if old is not None and new.dtype != torch.float32:
            self.expert_bias = new.float() if old.is_meta else old.to(new.device)
        return self

    def reset_draft_parameters(self, *, initializer_range: float) -> None:
        init.normal_(self.weight, mean=0.0, std=initializer_range)
        if self.expert_bias is not None:
            init.zeros_(self.expert_bias)

    def validate_draft_state(self) -> None:
        if self.config.expert_bias != (self.expert_bias is not None):
            raise ValueError("expert_bias buffer disagrees with the router node")
        if self.expert_bias is not None and (
            self.expert_bias.dtype != torch.float32
            or tuple(self.expert_bias.shape) != (self.num_experts,)
        ):
            raise ValueError("expert_bias must be fp32 with one entry per expert")

    def validate_export_state(self, state) -> None:
        expected = {"weight"} | ({"expert_bias"} if self.config.expert_bias else set())
        if set(state) != expected:
            raise ValueError(f"router state {sorted(state)} must be {sorted(expected)}")
        if self.config.expert_bias and (
            state["expert_bias"].dtype != torch.float32
            or tuple(state["expert_bias"].shape) != (self.num_experts,)
        ):
            raise ValueError("expert_bias must be fp32 with one entry per expert")

    def score(self, x: Tensor) -> Tensor:
        """`x` [T, H] -> fp32 affinities [T, E]; logits stay in the input dtype as in Qwen."""
        logits = F.linear(x, self.weight).float()
        return (
            logits.softmax(dim=-1)
            if self.config.scoring == "softmax"
            else torch.sigmoid(logits)
        )

    def select(self, scores: Tensor) -> Tensor:
        choice = scores if self.expert_bias is None else scores + self.expert_bias
        return choice.topk(self.top_k, dim=-1).indices

    def routing_weights(self, scores: Tensor, indices: Tensor) -> Tensor:
        weights = scores.gather(-1, indices)
        if self.config.normalize_top_k:
            # DeepSeek's 1e-20: a sigmoid top-k sum can be zero; a no-op for softmax.
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return weights * self.config.route_scale

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """`x` [T, H] -> fp32 scores [T, E], indices [T, K], fp32 weights [T, K]."""
        scores = self.score(x)
        indices = self.select(scores)
        return scores, indices, self.routing_weights(scores, indices)

    @torch.no_grad()
    def update_expert_bias(self, tokens_per_expert: Tensor, *, rate: float) -> None:
        """One DeepSeek-V3 step: nudge under-used experts up by `rate`, zero-mean overall."""
        if self.expert_bias is None or rate == 0:
            return
        load = tokens_per_expert.to(self.expert_bias.device, torch.float32)
        delta = rate * torch.sign(load.mean() - load)
        self.expert_bias += delta - delta.mean()


def grouped_mm_experts(
    x_sorted: Tensor,
    w_sorted: Tensor,
    counts: Tensor,
    *,
    gate_up_proj: Tensor,
    down_proj: Tensor,
    act_fn,
) -> Tensor:
    """Expert-sorted tokens [S, H] -> [S, H] via F.grouped_mm; `counts` [E] sizes the groups."""
    offsets = counts.cumsum(0, dtype=torch.int32)
    gate_up = F.grouped_mm(x_sorted, gate_up_proj.transpose(-1, -2), offs=offsets)
    gate, up = gate_up.chunk(2, dim=-1)
    # Weighting before the down GEMM keeps the [S, H] output out of autograd's saved tensors.
    hidden = (act_fn(gate) * up * w_sorted).to(gate_up.dtype)
    return F.grouped_mm(hidden, down_proj.transpose(-1, -2), offs=offsets)


def looped_experts(
    x_sorted: Tensor,
    w_sorted: Tensor,
    counts: Tensor,
    *,
    gate_up_proj: Tensor,
    down_proj: Tensor,
    act_fn,
) -> Tensor:
    """Same contract as grouped_mm_experts with one matmul pair per expert; any device."""
    outputs, start = [], 0
    for expert, count in enumerate(counts.tolist()):
        end = start + count
        gate_up = x_sorted[start:end] @ gate_up_proj[expert].T
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = (act_fn(gate) * up * w_sorted[start:end]).to(gate_up.dtype)
        outputs.append(hidden @ down_proj[expert].T)
        start = end
    return torch.cat(outputs)


# Routed-expert kernels over one parameter layout, picked by name once at startup.
EXPERT_KERNELS = {"grouped_mm": grouped_mm_experts, "loop": looped_experts}


class RoutedExperts(nn.Module):
    """Gated experts fused as gate_up_proj [E, 2I, H] and down_proj [E, H, I]."""

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        device=None,
        dtype=None,
    ):
        super().__init__()
        for name, value in (
            ("num_experts", num_experts),
            ("hidden_size", hidden_size),
            ("intermediate_size", intermediate_size),
        ):
            positive_int(name, value)
        self.num_experts = num_experts
        self.act_fn = ACT2FN[hidden_act]
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size,
                hidden_size,
                device=device,
                dtype=dtype,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
            )
        )

    def reset_draft_parameters(self, *, initializer_range: float) -> None:
        init.normal_(self.gate_up_proj, mean=0.0, std=initializer_range)
        init.normal_(self.down_proj, mean=0.0, std=initializer_range)

    def validate_export_state(self, state) -> None:
        expected = {
            "gate_up_proj": tuple(self.gate_up_proj.shape),
            "down_proj": tuple(self.down_proj.shape),
        }
        shapes = {name: tuple(tensor.shape) for name, tensor in state.items()}
        if shapes != expected:
            raise ValueError(f"expert state {shapes} must be {expected}")

    def forward(
        self, x: Tensor, indices: Tensor, weights: Tensor, *, kernel: str
    ) -> Tensor:
        """`x` [T, H], `indices` and `weights` [T, K] -> fp32 [T, H] summed over k."""
        num_tokens, top_k = indices.shape
        expert_ids, order = torch.sort(indices.reshape(-1), stable=True)
        # scatter_add_, not bincount: bincount reads the index range on the host.
        counts = expert_ids.new_zeros(self.num_experts).scatter_add_(
            0, expert_ids, torch.ones_like(expert_ids)
        )
        x_sorted = x[order // top_k]
        w_sorted = weights.reshape(-1)[order, None]
        if kernel not in EXPERT_KERNELS:
            raise ValueError(
                f"unknown experts kernel {kernel!r}; expected {sorted(EXPERT_KERNELS)}"
            )
        out = EXPERT_KERNELS[kernel](
            x_sorted,
            w_sorted,
            counts,
            gate_up_proj=self.gate_up_proj,
            down_proj=self.down_proj,
            act_fn=self.act_fn,
        )
        # index_copy_ is deterministic where index_add_ is not; sum the k slots in fp32.
        out = torch.empty_like(out).index_copy_(0, order, out)
        return out.view(num_tokens, top_k, -1).sum(dim=1, dtype=torch.float32)


class SharedExpert(nn.Module):
    """Always-on FFN added to the routed output: `gated` is act(gate_proj)·up_proj as in
    Qwen and DeepSeek, otherwise act(up_proj) as in Nemotron's relu2 MLP."""

    def __init__(
        self,
        config: MoESharedExpertSpec,
        *,
        hidden_size: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if type(config) is not MoESharedExpertSpec:
            raise TypeError("SharedExpert requires MoESharedExpertSpec")
        positive_int("hidden_size", hidden_size)
        self.config = config
        # Set by configure_kernels; fuses act(gate)·up, so the gated branch only.
        self.liger_swiglu = None
        self.act_fn = ACT2FN[config.hidden_act]
        width = config.intermediate_size
        self.gate_proj = (
            nn.Linear(hidden_size, width, bias=False, device=device, dtype=dtype)
            if config.gated
            else None
        )
        self.up_proj = nn.Linear(
            hidden_size, width, bias=False, device=device, dtype=dtype
        )
        self.down_proj = nn.Linear(
            width, hidden_size, bias=False, device=device, dtype=dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        up = self.up_proj(x)
        if self.gate_proj is None:
            return self.down_proj(self.act_fn(up))
        gate = self.gate_proj(x)
        activated = (
            self.liger_swiglu(gate, up)
            if self.liger_swiglu is not None
            else self.act_fn(gate) * up
        )
        return self.down_proj(activated)
