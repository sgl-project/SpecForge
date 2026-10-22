# coding=utf-8
"""Routed experts with their token dispatch, and the always-on shared expert."""

from __future__ import annotations

import abc
from typing import Type

import torch
import torch.nn.functional as F
from torch import nn

from .base import Registry, RoutingResult, TrainingState
from .config import MoEConfig

DISPATCH_MODES = ("sorted_loop", "grouped_mm")


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU in fp32: ``silu(gate) * up``."""
    return F.silu(gate.float()) * up.float()


class RoutedExperts(TrainingState, nn.Module, abc.ABC):
    """All ``n_routed_experts`` experts and the dispatch of tokens to them."""

    def __init__(self, cfg: MoEConfig, hidden_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.n_experts = cfg.n_routed_experts
        self.intermediate_size = cfg.moe_intermediate_size

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, routing: RoutingResult) -> torch.Tensor:
        """Weighted routed sum of ``x`` (``[T, hidden]``) as a fresh fp32 tensor."""

    @abc.abstractmethod
    def reset_parameters(self, std: float) -> None:
        """Initialize the expert weights with the draft's ``initializer_range``."""


EXPERTS_BACKENDS: Registry[Type[RoutedExperts]] = Registry("MoE experts backend")


def register_experts_backend(name: str):
    return EXPERTS_BACKENDS.register(name)


def build_routed_experts(cfg: MoEConfig, hidden_size: int) -> RoutedExperts:
    experts_cls = EXPERTS_BACKENDS.get(cfg.experts_backend)
    return experts_cls(cfg, hidden_size)


@register_experts_backend("grouped")
class GroupedExperts(RoutedExperts):
    """Stacked ``[E, out, in]`` expert weights with sorted-segment dispatch."""

    _WEIGHT_NAMES = ("w1", "w2", "w3")

    def __init__(self, cfg: MoEConfig, hidden_size: int) -> None:
        super().__init__(cfg, hidden_size)
        if cfg.dispatch not in DISPATCH_MODES:
            raise ValueError(
                f"unknown MoE dispatch {cfg.dispatch!r}; choose from {DISPATCH_MODES}"
            )
        # grouped_mm runs on CUDA when torch has it; the loop is the fallback.
        self.grouped_mm = cfg.dispatch == "grouped_mm" and hasattr(torch, "_grouped_mm")
        e, d, i = cfg.n_routed_experts, hidden_size, cfg.moe_intermediate_size
        # Stacked: grouped GEMMs read them directly; FSDP tracks 3 params, not 3*E.
        self.w1 = nn.Parameter(torch.empty(e, i, d))
        self.w2 = nn.Parameter(torch.empty(e, d, i))
        self.w3 = nn.Parameter(torch.empty(e, i, d))

    def reset_parameters(self, std: float) -> None:
        if self.w1.device.type == "meta":
            return
        for name in self._WEIGHT_NAMES:
            nn.init.normal_(getattr(self, name), mean=0.0, std=std)

    def forward(self, x: torch.Tensor, routing: RoutingResult) -> torch.Tensor:
        flat_expert = routing.indices.flatten()  # [T*k]
        # One stable argsort gives contiguous per-expert segments for both dispatches.
        order = flat_expert.argsort(stable=True)
        token_of = order // routing.topk  # routed token index per sorted slot
        x_sorted = x.index_select(0, token_of)
        w_sorted = routing.weights.reshape(-1, 1).index_select(0, order).float()
        counts = routing.counts
        y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)

        if self.grouped_mm and x.is_cuda:
            # No host sync; same math as the loop up to bf16 rounding.
            offs = counts.cumsum(0).to(torch.int32)
            gate = torch._grouped_mm(x_sorted, self.w1.transpose(-1, -2), offs=offs)
            up = torch._grouped_mm(x_sorted, self.w3.transpose(-1, -2), offs=offs)
            h = w_sorted * swiglu(gate, up)
            y_routed = torch._grouped_mm(
                h.to(x.dtype), self.w2.transpose(-1, -2), offs=offs
            )
            return y.index_add(0, token_of, y_routed.float())

        counts_list = counts.tolist()  # one host sync per MoE forward
        parts = []
        offset = 0
        for i, n in enumerate(counts_list):
            if n == 0:
                continue
            seg = x_sorted[offset : offset + n]
            h = w_sorted[offset : offset + n] * swiglu(
                F.linear(seg, self.w1[i]), F.linear(seg, self.w3[i])
            )
            parts.append(F.linear(h.to(seg.dtype), self.w2[i]))
            offset += n
        if not parts:
            return y
        return y.index_add(0, token_of, torch.cat(parts, dim=0).float())


# -- shared expert: an always-on SwiGLU FFN added to the routed output ----------


class SharedExpert(TrainingState, nn.Module, abc.ABC):
    """An always-on FFN, ``shared_expert_intermediate_size`` wide."""

    def __init__(self, cfg: MoEConfig, hidden_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.intermediate_size = cfg.shared_expert_intermediate_size

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is ``[T, hidden]``; return ``[T, hidden]`` in ``x.dtype``."""


SHARED_EXPERTS: Registry[Type[SharedExpert]] = Registry("MoE shared expert")


def register_shared_expert(name: str):
    return SHARED_EXPERTS.register(name)


def build_shared_expert(cfg: MoEConfig, hidden_size: int) -> SharedExpert:
    shared_cls = SHARED_EXPERTS.get(cfg.shared_expert)
    return shared_cls(cfg, hidden_size)


@register_shared_expert("swiglu")
class SwiGLUSharedExpert(SharedExpert):
    """``w2(swiglu(w1 x, w3 x))``, optionally scaled per token by ``sigmoid(gate(x))``."""

    def __init__(self, cfg: MoEConfig, hidden_size: int) -> None:
        super().__init__(cfg, hidden_size)
        width = cfg.shared_expert_intermediate_size
        # nn.Linear children are initialized by the HF _init_weights pass.
        self.w1 = nn.Linear(hidden_size, width, bias=False)
        self.w2 = nn.Linear(width, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, width, bias=False)
        self.gate = (
            nn.Linear(hidden_size, 1, bias=False)
            if cfg.shared_expert_gate == "sigmoid"
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = swiglu(self.w1(x), self.w3(x))
        y = self.w2(h.to(x.dtype))  # in the input dtype
        if self.gate is not None:
            y = torch.sigmoid(self.gate(x).float()).to(y.dtype) * y
        return y
