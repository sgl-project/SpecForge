# coding=utf-8
"""Routed experts: the expert weights and how tokens reach them.

:class:`RoutedExperts` is the contract; implementations register by name
(``MoEConfig.experts_backend``) and :class:`GroupedExperts` (``"grouped"``) is
the built-in one. An implementation owns the parameters of all
``n_routed_experts`` experts and returns the routed output as a fresh **fp32**
``[T, hidden]`` tensor: :class:`~.layer.MoELayer` adds the shared expert into
that accumulator and casts once, the order DeepSeek's reference
``MoE.forward`` uses (each expert's down projection runs in the model dtype;
their weighted sum and the shared expert are accumulated in fp32).

The weight layout is the implementation's choice; :mod:`.state_dict`
converters map it to the checkpoint files' naming.
"""

from __future__ import annotations

import abc
from typing import Type

import torch
import torch.nn.functional as F
from torch import nn

from .base import Registry, RoutingResult, TrainingState
from .config import MoEConfig


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
        """``x`` is ``[T, hidden]``; return the weighted routed sum ``[T, hidden]``
        as a fresh fp32 tensor."""

    @abc.abstractmethod
    def reset_parameters(self, std: float) -> None:
        """Initialize expert weights with the draft's ``initializer_range`` so an
        MoE FFN starts from the same distribution as the dense MLP it replaces."""


EXPERTS_BACKENDS: Registry[Type[RoutedExperts]] = Registry("MoE experts backend")


def register_experts_backend(name: str):
    return EXPERTS_BACKENDS.register(name)


def build_routed_experts(cfg: MoEConfig, hidden_size: int) -> RoutedExperts:
    return EXPERTS_BACKENDS.get(cfg.experts_backend)(cfg, hidden_size)


DISPATCH_MODES = ("sorted_loop", "grouped_mm")


@register_experts_backend("grouped")
class GroupedExperts(RoutedExperts):
    """All ``n_routed_experts`` experts as three stacked parameters
    ``w1``/``w2``/``w3`` of shape ``[E, out, in]`` with sorted-segment dispatch.

    Grouped GEMMs read the stacked tensors directly (a per-call ``torch.stack``
    of hundreds of expert weights would allocate a transient multi-GiB
    tensor), and FSDP ``use_orig_params`` tracks 3 tensors instead of ``3*E``.

    Dispatch (``MoEConfig.dispatch``): one stable argsort turns routing into
    contiguous per-expert segments, then

    - ``"sorted_loop"``: one small GEMM per active expert (one host sync per
      forward to read the segment sizes);
    - ``"grouped_mm"``: ``torch._grouped_mm`` over the segments with on-device
      offsets (no host sync). Used on CUDA when available; falls back to the
      loop elsewhere. Same math up to bf16 rounding.
    """

    _WEIGHT_NAMES = ("w1", "w2", "w3")

    def __init__(self, cfg: MoEConfig, hidden_size: int) -> None:
        super().__init__(cfg, hidden_size)
        if cfg.dispatch not in DISPATCH_MODES:
            raise ValueError(
                f"unknown MoE dispatch {cfg.dispatch!r}; choose from {DISPATCH_MODES}"
            )
        self.grouped_mm = cfg.dispatch == "grouped_mm" and hasattr(torch, "_grouped_mm")
        e, d, i = cfg.n_routed_experts, hidden_size, cfg.moe_intermediate_size
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
        order = flat_expert.argsort(stable=True)
        token_of = order // routing.topk  # routed token index per sorted slot
        x_sorted = x.index_select(0, token_of)
        w_sorted = routing.weights.reshape(-1, 1).index_select(0, order).float()
        counts = routing.counts
        y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)

        if self.grouped_mm and x.is_cuda:
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
