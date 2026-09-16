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
"""

from __future__ import annotations

import re

import torch
import torch.nn.functional as F
from torch import nn

from .config import MoEConfig
from .expert_parallel import draft_ep_layout, reduce_routed_output
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
        self.ep_group, ep_rank, self.ep_size = draft_ep_layout()
        if self.n_experts % self.ep_size:
            raise ValueError(
                f"n_routed_experts={self.n_experts} is not divisible by "
                f"expert_parallel_size={self.ep_size}"
            )
        local = self.n_experts // self.ep_size
        self.expert_start = ep_rank * local
        self.expert_end = self.expert_start + local
        # The stacked parameters hold only this rank's slice; expert_offset
        # travels with them so the checkpoint converter can restore the global
        # expert indices without knowing the topology.
        self.register_buffer(
            "expert_offset", torch.tensor(self.expert_start, dtype=torch.long)
        )
        if self.ep_size > 1:
            # Everything else in an MoE layer is replicated with identical
            # gradients; these three are the disjoint slice. The optimizer reads
            # the marker to keep the global gradient norm exact.
            self._specforge_rank_local_parameters = True
        e, d, i = local, hidden_size, self.intermediate_size
        self.w1 = nn.Parameter(torch.empty(e, i, d))
        self.w2 = nn.Parameter(torch.empty(e, d, i))
        self.w3 = nn.Parameter(torch.empty(e, i, d))

    @property
    def n_local_experts(self) -> int:
        return self.n_experts // self.ep_size

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Take this rank's slice out of a full stacked tensor.

        A checkpoint written by a different topology — or an unsharded one —
        carries all ``n_experts``; loading it into a sharded module would fail on
        shape. Slicing here keeps warm start and resume topology-agnostic.
        """
        if self.ep_size > 1:
            for name in self._WEIGHT_NAMES:
                key = prefix + name
                tensor = state_dict.get(key)
                if (
                    isinstance(tensor, torch.Tensor)
                    and tensor.dim() == 3
                    and tensor.shape[0] == self.n_experts
                ):
                    state_dict[key] = tensor[self.expert_start : self.expert_end]
        # expert_offset describes THIS rank's topology, not the checkpoint's. A
        # payload written by a different layout (or by an unsharded run, where it
        # is 0) must not move it, so the module's own value always wins.
        state_dict[prefix + "expert_offset"] = self.expert_offset
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def reset_parameters(self, std: float) -> None:
        if self.w1.device.type == "meta":
            return
        for name in self._WEIGHT_NAMES:
            nn.init.normal_(getattr(self, name), mean=0.0, std=std)

    def forward(self, x: torch.Tensor, routing: RoutingResult) -> torch.Tensor:
        flat_expert = routing.indices.flatten()  # [T*k]
        order = flat_expert.argsort(stable=True)
        counts = routing.counts

        if self.ep_size > 1:
            # Sorting by expert makes this rank's experts one contiguous run of
            # slots, so the slice is all it has to look at. Reading the bounds
            # costs the one host sync the sorted_loop path already pays; under
            # grouped_mm it is the price of expert parallelism.
            counts_list = counts.tolist()
            start_slot = sum(counts_list[: self.expert_start])
            local_counts = counts_list[self.expert_start : self.expert_end]
            order = order[start_slot : start_slot + sum(local_counts)]
        else:
            local_counts = None

        token_of = order // routing.topk  # routed token index per sorted slot
        x_sorted = x.index_select(0, token_of)
        w_sorted = routing.weights.reshape(-1, 1).index_select(0, order).float()

        if self.grouped_mm and x.is_cuda:
            offs = (
                counts.cumsum(0).to(torch.int32)
                if local_counts is None
                else torch.tensor(local_counts, dtype=torch.long, device=x.device)
                .cumsum(0)
                .to(torch.int32)
            )
            gate = torch._grouped_mm(x_sorted, self.w1.transpose(-1, -2), offs=offs)
            up = torch._grouped_mm(x_sorted, self.w3.transpose(-1, -2), offs=offs)
            h = w_sorted * swiglu_clamped(gate, up, self.swiglu_limit)
            y_routed = torch._grouped_mm(
                h.to(x.dtype), self.w2.transpose(-1, -2), offs=offs
            )
        else:
            if local_counts is None:
                local_counts = counts.tolist()  # one host sync per MoE forward
            parts = []
            offset = 0
            for i, n in enumerate(local_counts):
                if n == 0:
                    continue
                seg = x_sorted[offset : offset + n]
                h = w_sorted[offset : offset + n] * swiglu_clamped(
                    F.linear(seg, self.w1[i]),
                    F.linear(seg, self.w3[i]),
                    self.swiglu_limit,
                )
                parts.append(F.linear(h.to(seg.dtype), self.w2[i]))
                offset += n
            if parts:
                y_routed = torch.cat(parts, dim=0)
            else:
                # No token routed here. Under EP that is routine on an imbalanced
                # route, and the rank must still reach the collective below with
                # every expert parameter attached to the graph, or DDP and the
                # collective order diverge between ranks.
                y_routed = None

        y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        if self.ep_size > 1:
            # A rank owning no selected expert for this microbatch is routine on
            # an imbalanced route, and without this its input and the routing
            # weights would have no differentiable consumer at all: autograd
            # would never reach the expert-parallel seam in MoELayer, its
            # all-reduce would never be issued, and the peers that did issue
            # theirs would hang. The zero terms keep the graph -- and with it the
            # collective order -- identical on every rank, and add nothing to the
            # value. The expert parameters are attached for the same reason, so
            # DDP still sees every registered parameter.
            y = (
                y
                + x.float() * 0.0
                + routing.weights.float().sum() * 0.0
                + sum(
                    getattr(self, name).reshape(-1)[0].float() * 0.0
                    for name in self._WEIGHT_NAMES
                )
            )
        if y_routed is not None:
            y = y.index_add(0, token_of, y_routed.float())
        if self.ep_size > 1:
            y = reduce_routed_output(y, self.ep_group)
        return y.to(x.dtype)


_STACKED_KEY = re.compile(r"^(?P<base>(?:.*\.)?experts)\.(?P<w>w[123])$")
_OFFSET_KEY = re.compile(r"^(?P<base>(?:.*\.)?experts)\.expert_offset$")
_PER_EXPERT_KEY = re.compile(
    r"^(?P<base>(?:.*\.)?experts)\.(?P<idx>\d+)\.(?P<w>w[123])\.weight$"
)


def _expert_offsets(state: dict) -> dict:
    """``base`` -> global index of its first stacked expert (0 when unsharded)."""
    offsets = {}
    for key, value in state.items():
        m = _OFFSET_KEY.match(key)
        if m is not None:
            offsets[m["base"]] = int(value)
    return offsets


def unstack_grouped_expert_state_dict(state: dict) -> dict:
    """``experts.w1`` [E, out, in] -> ``experts.{i}.w1.weight``; no-op otherwise.

    Under expert parallelism the stacked tensor holds only this rank's slice, and
    ``experts.expert_offset`` says where the slice starts. The checkpoint keeps
    global expert indices and drops the offset: the file is then the same shape
    of thing at every topology, and the indices alone say what a shard contains.
    """
    offsets = _expert_offsets(state)
    out = {}
    for key, value in state.items():
        if _OFFSET_KEY.match(key) is not None:
            continue
        m = _STACKED_KEY.match(key)
        if m is None or not isinstance(value, torch.Tensor) or value.dim() != 3:
            out[key] = value
            continue
        base = m["base"]
        offset = offsets.get(base, 0)
        for i in range(value.shape[0]):
            out[f"{base}.{offset + i}.{m['w']}.weight"] = value[i]
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
        present = sorted(members)
        first, last = present[0], present[-1]
        if present != list(range(first, last + 1)):
            raise KeyError(
                f"{base}.*.{w}.weight is missing expert indices: have {present}"
            )
        out[f"{base}.{w}"] = torch.stack(
            [members[i] for i in range(first, last + 1)], dim=0
        )
        if first:
            # A single expert-parallel shard rather than the whole layer; the
            # module reads this back to know which experts it is holding.
            out[f"{base}.expert_offset"] = torch.tensor(first, dtype=torch.long)
    return out


register_state_dict_converter(
    "grouped_experts",
    to_checkpoint=unstack_grouped_expert_state_dict,
    from_checkpoint=stack_grouped_expert_state_dict,
)
