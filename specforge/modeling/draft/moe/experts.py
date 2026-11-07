# coding=utf-8
"""Routed and shared SwiGLU experts for draft MoE layers."""

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

Activation = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
# Other backends, such as Ascend NPU and XPU, take the per-expert loop.
_GROUPED_MM_DEVICES = ("cuda", "cpu")


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up``: the default expert activation."""
    return F.silu(gate) * up


def grouped_mm_experts(
    x: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    activation: Activation,
) -> torch.Tensor:
    """Sum each token's routing-weighted top-k expert outputs, in fp32."""
    num_tokens, top_k = indices.shape
    expert_ids, order = torch.sort(indices.reshape(-1), stable=True)
    counts = torch.zeros(gate_up_proj.shape[0], dtype=torch.long, device=x.device)
    counts.scatter_add_(0, expert_ids, torch.ones_like(expert_ids))
    x_sorted = x[order // top_k]
    w_sorted = weights.reshape(-1)[order, None]
    # grouped_mm needs 16-byte aligned rows.
    aligned = x.shape[-1] % 8 == 0 and down_proj.shape[-1] % 8 == 0
    if x.device.type in _GROUPED_MM_DEVICES and aligned:
        offsets = counts.cumsum(0, dtype=torch.int32)
        gate_up = F.grouped_mm(x_sorted, gate_up_proj.transpose(-1, -2), offs=offsets)
        # Weighting before the down GEMM keeps [S, H] out of autograd's saved tensors.
        hidden = (activation(*gate_up.chunk(2, dim=-1)) * w_sorted).to(gate_up.dtype)
        out = F.grouped_mm(hidden, down_proj.transpose(-1, -2), offs=offsets)
    else:
        out = _looped_experts(
            x_sorted, w_sorted, counts, gate_up_proj, down_proj, activation
        )
    # Unsort and sum each token's k slots (index_add_ is nondeterministic on CUDA).
    out = torch.empty_like(out).index_copy_(0, order, out)
    return out.view(num_tokens, top_k, out.shape[-1]).sum(dim=1, dtype=torch.float32)


def _looped_experts(
    x_sorted: torch.Tensor,
    w_sorted: torch.Tensor,
    counts: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    activation: Activation,
) -> torch.Tensor:
    """The grouped GEMMs above as one matmul pair per expert."""
    outputs = []
    start = 0
    for expert, count in enumerate(counts.tolist()):
        end = start + count
        gate_up = x_sorted[start:end] @ gate_up_proj[expert].T
        hidden = activation(*gate_up.chunk(2, dim=-1)) * w_sorted[start:end]
        outputs.append(hidden.to(gate_up.dtype) @ down_proj[expert].T)
        start = end
    return torch.cat(outputs)


class RoutedExperts(nn.Module):
    """SwiGLU experts in the stacked Qwen3.5-MoE layout ([E, 2I, H], [E, H, I])."""

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        initializer_range: float,
        activation: Activation = swiglu,
    ) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        self.activation = activation
        self.reset_parameters(initializer_range)

    def reset_parameters(self, std: float) -> None:
        # HF's default _init_weights skips bare Parameters.
        nn.init.normal_(self.gate_up_proj, std=std)
        nn.init.normal_(self.down_proj, std=std)

    def forward(
        self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """``x`` [T, H], ``indices`` and ``weights`` [T, K] -> fp32 [T, H]."""
        return grouped_mm_experts(
            x, indices, weights, self.gate_up_proj, self.down_proj, self.activation
        )


class SharedExpert(nn.Module):
    """An always-on SwiGLU FFN that the MoE layer adds to the routed output."""

    def __init__(
        self, hidden_size: int, intermediate_size: int, activation: Activation = swiglu
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        return self.down_proj(self.activation(gate, self.up_proj(x)).to(gate.dtype))
