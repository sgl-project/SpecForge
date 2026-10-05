# coding=utf-8
"""Routed experts as three stacked parameters with sorted-segment dispatch.

Weights live as ``w1``/``w2``/``w3`` of shape ``[E, out, in]``: grouped GEMMs
read them directly (a per-call ``torch.stack`` of hundreds of expert weights
would allocate a transient multi-GiB tensor) and FSDP ``use_orig_params``
tracks 3 tensors instead of ``3*E``. Checkpoint FILES keep the official
per-expert naming (``experts.{i}.w{1,2,3}.weight``) through the converter
registered below.

Dispatch (``MoEConfig.dispatch``):

- ``"sorted_loop"``: one stable argsort turns routing into contiguous
  per-expert segments, then one small GEMM per active expert. A per-expert
  ``torch.where`` loop scales launch and autograd overhead with the number of
  ACTIVE experts (~2x step time once the balancer spreads load).
- ``"grouped_mm"``: the same segments through ``torch._grouped_mm`` with
  on-device offsets (no host sync). Used on CUDA when available; falls back to
  the loop elsewhere. Same math up to bf16 rounding.

Expert parallelism (:mod:`.expert_parallel`): after
:meth:`GroupedExperts.apply_expert_parallel` the stacked parameters are
``DTensor`` ``Shard(0)`` slices over the ``ep`` mesh and the forward receives the
EP group's gathered tokens. Sorting by expert makes this rank's experts one
contiguous run of slots, so the local work is a slice of the sorted order; the
returned fp32 output is this rank's PARTIAL sum, which :class:`MoELayer`
reduce-scatters across the group.
"""

from __future__ import annotations

import re
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from .config import MoEConfig
from .expert_parallel import (
    ExpertParallelLayout,
    expert_parallel_layout,
    local_expert_weight,
    shard_expert_parameter,
)
from .experts import RoutedExperts, register_experts_backend
from .router import RoutingResult
from .state_dict import register_state_dict_converter

DISPATCH_MODES = ("sorted_loop", "grouped_mm")


def swiglu_clamped(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """SwiGLU in fp32 with the DeepSeek-V4 activation clamp (``limit`` 0 = off)."""
    gate = gate.float()
    up = up.float()
    if limit > 0:
        up = torch.clamp(up, min=-limit, max=limit)
        gate = torch.clamp(gate, max=limit)
    return F.silu(gate) * up


@register_experts_backend("grouped")
class GroupedExperts(RoutedExperts):
    _WEIGHT_NAMES = ("w1", "w2", "w3")

    def __init__(self, cfg: MoEConfig, hidden_size: int) -> None:
        super().__init__(cfg, hidden_size)
        if cfg.dispatch not in DISPATCH_MODES:
            raise ValueError(
                f"unknown MoE dispatch {cfg.dispatch!r}; choose from {DISPATCH_MODES}"
            )
        self.grouped_mm = cfg.dispatch == "grouped_mm" and hasattr(torch, "_grouped_mm")
        self.swiglu_limit = float(cfg.swiglu_limit)
        e, d, i = self.n_experts, hidden_size, self.intermediate_size
        self.w1 = nn.Parameter(torch.empty(e, i, d))
        self.w2 = nn.Parameter(torch.empty(e, d, i))
        self.w3 = nn.Parameter(torch.empty(e, i, d))
        #: Set by :meth:`apply_expert_parallel`; ``None`` keeps every expert local.
        self.ep: Optional[ExpertParallelLayout] = None

    @property
    def n_local_experts(self) -> int:
        return self.ep.n_local_experts if self.ep is not None else self.n_experts

    def apply_expert_parallel(self, ep_mesh: DeviceMesh) -> ExpertParallelLayout:
        """Keep only this rank's ``[E/ep]`` slice of the stacked experts.

        Call after initialization and warm start (the full tensors are sliced
        in place) and before FSDP wrapping, which shards the slice further.
        """
        if self.ep is not None:
            raise RuntimeError(
                "expert parallelism was already applied to these experts"
            )
        layout = expert_parallel_layout(ep_mesh, self.n_experts)
        for name in self._WEIGHT_NAMES:
            setattr(self, name, shard_expert_parameter(getattr(self, name), ep_mesh))
        self.ep = layout
        return layout

    def reset_parameters(self, std: float) -> None:
        if self.w1.device.type == "meta":
            return
        for name in self._WEIGHT_NAMES:
            nn.init.normal_(local_expert_weight(getattr(self, name)), mean=0.0, std=std)

    def forward(self, x: torch.Tensor, routing: RoutingResult) -> torch.Tensor:
        flat_expert = routing.indices.flatten()  # [T*k]
        order = flat_expert.argsort(stable=True)
        counts = routing.counts
        ep = self.ep
        w1, w2, w3 = (local_expert_weight(getattr(self, n)) for n in self._WEIGHT_NAMES)

        counts_list = None
        n_local_tokens = None
        if ep is None:
            order_local = order
            local_counts = counts
        else:
            # Sorting by expert makes this rank's experts one contiguous run of
            # slots; reading its bounds is the one host sync EP costs per MoE
            # layer (the sorted_loop path pays it anyway).
            counts_list = counts.tolist()
            start_slot = sum(counts_list[: ep.expert_start])
            n_local_tokens = sum(counts_list[ep.expert_start : ep.expert_end])
            order_local = order[start_slot : start_slot + n_local_tokens]
            local_counts = counts[ep.expert_start : ep.expert_end]

        token_of = order_local // routing.topk  # routed token index per sorted slot
        x_sorted = x.index_select(0, token_of)
        w_sorted = routing.weights.reshape(-1, 1).index_select(0, order_local).float()

        y_routed = None
        if n_local_tokens == 0:
            # No token routed to this rank's experts this micro-batch: routine
            # on an imbalanced route under EP. The zero terms below keep the
            # graph (and so the collective order) identical on every rank.
            pass
        elif self.grouped_mm and x.is_cuda:
            offs = local_counts.cumsum(0).to(torch.int32)
            gate = torch._grouped_mm(x_sorted, w1.transpose(-1, -2), offs=offs)
            up = torch._grouped_mm(x_sorted, w3.transpose(-1, -2), offs=offs)
            h = w_sorted * swiglu_clamped(gate, up, self.swiglu_limit)
            y_routed = torch._grouped_mm(h.to(x.dtype), w2.transpose(-1, -2), offs=offs)
        else:
            if counts_list is None:
                counts_list = counts.tolist()  # one host sync per MoE forward
            local_list = (
                counts_list
                if ep is None
                else counts_list[ep.expert_start : ep.expert_end]
            )
            parts = []
            offset = 0
            for i, n in enumerate(local_list):
                if n == 0:
                    continue
                seg = x_sorted[offset : offset + n]
                h = w_sorted[offset : offset + n] * swiglu_clamped(
                    F.linear(seg, w1[i]),
                    F.linear(seg, w3[i]),
                    self.swiglu_limit,
                )
                parts.append(F.linear(h.to(seg.dtype), w2[i]))
                offset += n
            if parts:
                y_routed = torch.cat(parts, dim=0)

        y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        if ep is not None:
            # Keep the gathered input, the combine weights and every expert
            # parameter on the graph even when this rank computed nothing:
            # autograd must reach the EP collectives and FSDP2 must see the same
            # set of gradients on every rank. Adds nothing to the value.
            y = (
                y
                + x.reshape(-1)[0].float() * 0.0
                + routing.weights.float().sum() * 0.0
                + sum(w.reshape(-1)[0].float() * 0.0 for w in (w1, w2, w3))
            )
        if y_routed is not None:
            y = y.index_add(0, token_of, y_routed.float())
        if ep is not None:
            # This rank's partial sum in fp32; MoELayer reduce-scatters it over
            # the EP group and casts afterwards.
            return y
        return y.to(x.dtype)


_STACKED_KEY = re.compile(r"^(?P<base>(?:.*\.)?experts)\.(?P<w>w[123])$")
_PER_EXPERT_KEY = re.compile(
    r"^(?P<base>(?:.*\.)?experts)\.(?P<idx>\d+)\.(?P<w>w[123])\.weight$"
)


def unstack_grouped_expert_state_dict(state: dict) -> dict:
    """``experts.w1`` [E, out, in] -> ``experts.{i}.w1.weight``; no-op otherwise."""
    out = {}
    for key, value in state.items():
        m = _STACKED_KEY.match(key)
        if m is None or not isinstance(value, torch.Tensor) or value.dim() != 3:
            out[key] = value
            continue
        if isinstance(value, DTensor):
            raise TypeError(
                f"{key} is still an expert-parallel DTensor shard; gather the full "
                "state (get_model_state_dict(full_state_dict=True)) before converting"
            )
        for i in range(value.shape[0]):
            out[f"{m['base']}.{i}.{m['w']}.weight"] = value[i]
    return out


def stack_grouped_expert_state_dict(state: dict) -> dict:
    """Inverse of :func:`unstack_grouped_expert_state_dict`."""
    groups: dict = {}
    out = {}
    for key, value in state.items():
        m = _PER_EXPERT_KEY.match(key)
        if m is None:
            out[key] = value
            continue
        groups.setdefault((m["base"], m["w"]), {})[int(m["idx"])] = value
    for (base, w), members in groups.items():
        n = max(members) + 1
        if sorted(members) != list(range(n)):
            raise KeyError(
                f"{base}.*.{w}.weight is missing expert indices: have {sorted(members)}"
            )
        out[f"{base}.{w}"] = torch.stack([members[i] for i in range(n)], dim=0)
    return out


register_state_dict_converter(
    "grouped_experts",
    to_checkpoint=unstack_grouped_expert_state_dict,
    from_checkpoint=stack_grouped_expert_state_dict,
)
