"""Routed experts over fused weights with named kernels, and the always-on shared expert."""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import initialization as init
from transformers.activations import ACT2FN

from specforge.config.draft._validation import positive_int
from specforge.config.draft.mlp import MoESharedExpertSpec

# Routed-expert kernels over one parameter layout, picked by name once at startup.
EXPERT_KERNELS: dict[str, Callable[..., Tensor]] = {}


def expert_kernel(name: str):
    """Register the decorated routed-expert kernel under `name`."""

    def register(kernel):
        if name in EXPERT_KERNELS:
            raise ValueError(f"experts kernel {name!r} already registered")
        EXPERT_KERNELS[name] = kernel
        return kernel

    return register


@expert_kernel("grouped_mm")
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


@expert_kernel("loop")
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

    @torch.no_grad()
    def probe_kernel(self, kernel: str) -> None:
        """Run `kernel` once on zeros routed to expert 0, so a kernel the installed
        torch cannot run on this device, dtype or width fails at startup."""
        x = self.gate_up_proj.new_zeros(16, self.gate_up_proj.shape[-1])
        indices = torch.zeros(16, 1, dtype=torch.long, device=x.device)
        weights = torch.ones(16, 1, dtype=torch.float32, device=x.device)
        self(x, indices, weights, kernel=kernel)


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
