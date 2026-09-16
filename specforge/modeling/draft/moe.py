# coding=utf-8
"""DeepSeek-V4-style sparse MoE FFN for draft models.

sqrtsoftplus top-k routing with the aux-loss-free balancing bias, one shared
expert, and the routed experts stored as three stacked parameters ([E, out,
in]) dispatched over sorted per-expert segments. Checkpoint FILES keep the
official per-expert naming (``experts.{i}.w{1,2,3}.weight``); the module's
parameters keep the stacked layout FSDP ``use_orig_params`` requires — the
``unstack``/``stack`` helpers convert at the save/load boundary.

Configured by these draft-config fields: ``hidden_size``,
``n_routed_experts``, ``num_experts_per_tok``, ``moe_intermediate_size``,
``n_shared_experts`` (must be 1), ``scoring_func`` (``"sqrtsoftplus"``),
``routed_scaling_factor``, ``swiglu_limit``, and
``dflash_config["moe_grouped_dispatch"]``.
"""

from __future__ import annotations

import math
import re
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .expert_parallel import (
    CopyToExpertParallel,
    draft_ep_layout,
    reduce_replicated_gradients,
    reduce_routed_output,
)


class MoEExpert(nn.Module):
    """One SwiGLU expert with the optional DeepSeek-V4 activation clamp."""

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(
        self, x: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class MoEGate(nn.Module):
    """sqrtsoftplus top-k gate with the aux-loss-free balancing bias.

    ``bias`` shifts scores for expert selection only (never the routing
    weights) and is updated by a sign controller from all-reduced expert
    loads, not by gradients — so it lives as an fp32 buffer that survives
    module-wide dtype casts."""

    def __init__(self, config):
        super().__init__()
        if config.scoring_func != "sqrtsoftplus":
            raise ValueError(
                f"MoEGate implements sqrtsoftplus scoring only, got "
                f"{config.scoring_func!r}"
            )
        self.topk = config.num_experts_per_tok
        self.route_scale = config.routed_scaling_factor
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size)
        )
        self.register_buffer(
            "bias", torch.zeros(config.n_routed_experts, dtype=torch.float32)
        )

    def _apply(self, fn, recurse=True):
        module = super()._apply(fn, recurse)
        # The sign-controller steps (~1e-3) vanish under bf16 rounding once
        # bias magnitudes grow; keep the buffer fp32 through .to(dtype) casts.
        if module.bias.dtype != torch.float32:
            module.bias.data = module.bias.data.float()
        return module

    def forward(self, x: torch.Tensor):
        scores = F.linear(x.float(), self.weight.float())
        scores = F.softplus(scores).sqrt()
        indices = (scores + self.bias).topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights, indices


class GroupedExperts(nn.Module):
    """The routed experts as three stacked parameters ([E, out, in]).

    Grouped-GEMM dispatch reads the stacked parameters directly (a per-call
    ``torch.stack`` of hundreds of weights allocates a transient multi-GiB
    tensor), and FSDP ``use_orig_params`` tracks 3 view tensors instead of
    3*E. State-dict hooks accept the official per-expert naming
    (``experts.{i}.w{1,2,3}.weight``) so checkpoints, warm-start exports,
    and bundlers are unaffected.
    """

    _WEIGHT_NAMES = ("w1", "w2", "w3")

    def __init__(self, n_experts: int, dim: int, inter_dim: int, swiglu_limit: float):
        super().__init__()
        self.n_experts = n_experts
        self.swiglu_limit = swiglu_limit
        self.ep_group, ep_rank, self.ep_size = draft_ep_layout()
        if n_experts % self.ep_size:
            raise ValueError(
                f"n_routed_experts={n_experts} is not divisible by "
                f"expert_parallel_size={self.ep_size}"
            )
        self.n_local_experts = n_experts // self.ep_size
        self.expert_start = ep_rank * self.n_local_experts
        self.expert_end = self.expert_start + self.n_local_experts
        # The stacked parameters hold only this rank's slice; expert_offset
        # travels with them so the checkpoint converter can restore the global
        # expert indices without knowing the topology.
        self.register_buffer(
            "expert_offset", torch.tensor(self.expert_start, dtype=torch.long)
        )
        if self.ep_size > 1:
            # Everything else in the MoE is replicated with identical gradients;
            # these three are the disjoint slice. The optimizer reads the marker
            # to keep the global gradient norm exact.
            self._specforge_rank_local_parameters = True
        local = self.n_local_experts
        self.w1 = nn.Parameter(torch.empty(local, inter_dim, dim))
        self.w2 = nn.Parameter(torch.empty(local, dim, inter_dim))
        self.w3 = nn.Parameter(torch.empty(local, inter_dim, dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.w1.device.type == "meta":
            return
        for name in self._WEIGHT_NAMES:
            stacked = getattr(self, name)
            for i in range(self.n_local_experts):
                # per-expert slices match nn.Linear's default init exactly
                nn.init.kaiming_uniform_(stacked[i], a=math.sqrt(5))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Native stacked keys (``experts.w1`` [E, out, in]) load through the
        # stock path; that is also what FSDP's use_orig_params state-dict
        # hooks require (they assert the module's own parameter FQNs exist).
        # expert_offset describes THIS rank's topology, not the checkpoint's;
        # a payload written by another layout must never move it.
        state_dict[f"{prefix}expert_offset"] = self.expert_offset
        if any(f"{prefix}{name}" in state_dict for name in self._WEIGHT_NAMES):
            if self.ep_size > 1:
                for name in self._WEIGHT_NAMES:
                    key = f"{prefix}{name}"
                    tensor = state_dict.get(key)
                    if (
                        isinstance(tensor, torch.Tensor)
                        and tensor.dim() == 3
                        and tensor.shape[0] == self.n_experts
                    ):
                        state_dict[key] = tensor[self.expert_start : self.expert_end]
            return super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
        # Official per-expert naming: warm starts and trainer checkpoints
        # (see unstack_grouped_expert_state_dict).
        consumed = {f"{prefix}expert_offset"}
        for name in self._WEIGHT_NAMES:
            stacked = getattr(self, name)
            # Files name experts globally; this rank fills its own slice.
            for local, expert in enumerate(range(self.expert_start, self.expert_end)):
                key = f"{prefix}{expert}.{name}.weight"
                if key not in state_dict:
                    missing_keys.append(key)
                    continue
                consumed.add(key)
                value = state_dict[key]
                if tuple(value.shape) != tuple(stacked[local].shape):
                    error_msgs.append(
                        f"size mismatch for {key}: checkpoint "
                        f"{tuple(value.shape)}, model {tuple(stacked[local].shape)}"
                    )
                    continue
                with torch.no_grad():
                    stacked[local].copy_(value)
        if strict:
            for key in state_dict:
                if key.startswith(prefix) and key not in consumed:
                    # Experts owned by other ranks are expected to be here when a
                    # whole-layer checkpoint is loaded into one shard.
                    if self.ep_size > 1 and _PER_EXPERT_KEY.match(key):
                        continue
                    unexpected_keys.append(key)


_STACKED_EXPERT_KEY = re.compile(r"^(?P<base>(?:.*\.)?experts)\.(?P<w>w[123])$")
_EXPERT_OFFSET_KEY = re.compile(r"^(?P<base>(?:.*\.)?experts)\.expert_offset$")
_PER_EXPERT_KEY = re.compile(
    r"^(?P<base>(?:.*\.)?experts)\.(?P<idx>\d+)\.(?P<w>w[123])\.weight$"
)


def unstack_grouped_expert_state_dict(state: dict) -> dict:
    """Rewrite native stacked expert tensors into the official per-expert
    naming (``experts.w1`` [E, out, in] -> ``experts.{i}.w1.weight``).

    A no-op for state dicts without grouped experts.

    Under expert parallelism the stacked tensor holds only this rank's slice and
    ``experts.expert_offset`` says where it starts. The file keeps global expert
    indices and drops the offset, so a shard is described by its indices alone
    and reads the same at any topology.
    """
    offsets = {
        m["base"]: int(value)
        for key, value in state.items()
        if (m := _EXPERT_OFFSET_KEY.match(key)) is not None
    }
    out = {}
    for key, value in state.items():
        if _EXPERT_OFFSET_KEY.match(key) is not None:
            continue
        m = _STACKED_EXPERT_KEY.match(key)
        if m is None or not isinstance(value, torch.Tensor) or value.dim() != 3:
            out[key] = value
            continue
        base = m["base"]
        offset = offsets.get(base, 0)
        for i in range(value.shape[0]):
            out[f"{base}.{offset + i}.{m['w']}.weight"] = value[i]
    return out


def stack_grouped_expert_state_dict(state: dict) -> dict:
    """Inverse of :func:`unstack_grouped_expert_state_dict`: group official
    per-expert keys back into native stacked tensors before loading into a
    module (or an FSDP wrapper) that owns stacked parameters."""
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
            # One expert-parallel shard rather than the whole layer.
            out[f"{base}.expert_offset"] = torch.tensor(first, dtype=torch.long)
    return out


class SparseMoE(nn.Module):
    """Routed MoE FFN: sorted-segment dispatch, optional grouped GEMMs."""

    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        # Grouped dispatch runs the three expert linears as grouped GEMMs
        # over the sorted segments (identical math, bf16-rounding-level
        # output differences). Opt-in via dflash_config.
        self.grouped_dispatch = bool(
            (getattr(config, "dflash_config", None) or {}).get(
                "moe_grouped_dispatch", False
            )
        ) and hasattr(torch, "_grouped_mm")
        self.gate = MoEGate(config)
        self.experts = GroupedExperts(
            config.n_routed_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            config.swiglu_limit,
        )
        if config.n_shared_experts != 1:
            raise ValueError("SparseMoE requires exactly one shared expert")
        self.shared_experts = MoEExpert(
            config.hidden_size, config.moe_intermediate_size, config.swiglu_limit
        )
        self.bias_update_rate = 0.0
        self.last_expert_load: Optional[torch.Tensor] = None
        self._pending_counts: Optional[torch.Tensor] = None
        self.ep_size = self.experts.ep_size
        if self.ep_size > 1:
            # The gate is replicated, but this rank's loss only reaches the
            # combine weights of the experts it owns, so its weight gradient
            # here is a partial sum of the real one.
            reduce_replicated_gradients(self.gate, self.experts.ep_group)

    def apply_pending_balance_update(self) -> None:
        """noaux_tc balancing: identical on every rank via all-reduced loads.

        Called from the MODEL forward, outside any activation-checkpoint
        region: mutating gate.bias inside the stage forward would make the
        checkpoint recompute route differently (different segment shapes ->
        CheckpointError) and re-fire the all_reduce during backward.
        """
        import torch.distributed as dist

        counts = self._pending_counts
        self._pending_counts = None
        if counts is None or self.bias_update_rate <= 0:
            return
        counts = counts.float()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts)
        self.last_expert_load = counts
        error = counts.mean() - counts
        with torch.no_grad():
            self.gate.bias += self.bias_update_rate * torch.sign(error)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        # Under expert parallelism the tokens are replicated and the experts are
        # not, so the gate and the experts each hold a PARTIAL dL/dx: the gate
        # because only the slots of locally owned experts carry gradient, the
        # experts because this rank ran only its own. The seam sums both. The
        # shared expert stays outside it -- it is replicated and already sees the
        # whole loss, and reducing it would multiply it by the group size.
        routed_x = (
            CopyToExpertParallel.apply(x, self.experts.ep_group)
            if self.ep_size > 1
            else x
        )
        weights, indices = self.gate(routed_x)
        # scatter_add instead of bincount: CUDA bincount hides a device sync.
        flat_indices = indices.flatten()
        counts = torch.zeros(
            self.n_routed_experts, dtype=torch.long, device=flat_indices.device
        ).scatter_add_(0, flat_indices, torch.ones_like(flat_indices))
        if self.training:
            # Overwrite (never accumulate): a checkpoint recompute re-runs
            # this forward and must leave identical state behind.
            self._pending_counts = counts

        # Sorted dispatch: one argsort turns routing into contiguous
        # per-expert segments. A per-expert `torch.where(indices == i)` loop
        # scales its kernel-launch and autograd overhead with the number of
        # ACTIVE experts — ~2x step time once the balancer spreads load.
        flat_expert = indices.flatten()  # [T*k]
        order = flat_expert.argsort(stable=True)
        local_counts = None
        if self.ep_size > 1:
            # Sorting by expert makes this rank's experts one contiguous run of
            # slots, so the slice is all it has to look at.
            counts_list = counts.tolist()
            experts = self.experts
            start_slot = sum(counts_list[: experts.expert_start])
            local_counts = counts_list[experts.expert_start : experts.expert_end]
            order = order[start_slot : start_slot + sum(local_counts)]
        token_of = order // self.gate.topk  # routed token index per slot
        x_sorted = routed_x.index_select(0, token_of)
        w_sorted = weights.reshape(-1, 1).index_select(0, order).to(torch.float32)

        e = self.experts
        limit = e.swiglu_limit
        if self.grouped_dispatch and x.is_cuda:
            # Three grouped GEMMs over the stacked parameters; segment
            # offsets stay on device, so no host sync in this path.
            offs = (
                counts.cumsum(0).to(torch.int32)
                if local_counts is None
                else torch.tensor(local_counts, dtype=torch.long, device=x.device)
                .cumsum(0)
                .to(torch.int32)
            )
            gate_h = torch._grouped_mm(
                x_sorted, e.w1.transpose(-1, -2), offs=offs
            ).float()
            up = torch._grouped_mm(x_sorted, e.w3.transpose(-1, -2), offs=offs).float()
            if limit > 0:
                up = torch.clamp(up, min=-limit, max=limit)
                gate_h = torch.clamp(gate_h, max=limit)
            h = w_sorted * (F.silu(gate_h) * up)
            y_routed = torch._grouped_mm(
                h.to(x.dtype), e.w2.transpose(-1, -2), offs=offs
            )
            y = torch.zeros_like(x, dtype=torch.float32)
            if self.ep_size > 1:
                y = y + routed_x.float() * 0.0 + weights.float().sum() * 0.0
            y = y.index_add(0, token_of, y_routed.float())
            if self.ep_size > 1:
                y = reduce_routed_output(y, self.experts.ep_group)
            y = y + self.shared_experts(x)
            return y.to(x.dtype).view(shape)

        if local_counts is None:
            local_counts = counts.tolist()  # one host sync per MoE forward

        y_parts = []
        offset = 0
        for i, n in enumerate(local_counts):
            if n == 0:
                continue
            seg = x_sorted[offset : offset + n]
            gate_h = F.linear(seg, e.w1[i]).float()
            up = F.linear(seg, e.w3[i]).float()
            if limit > 0:
                up = torch.clamp(up, min=-limit, max=limit)
                gate_h = torch.clamp(gate_h, max=limit)
            h = w_sorted[offset : offset + n] * (F.silu(gate_h) * up)
            y_parts.append(F.linear(h.to(seg.dtype), e.w2[i]))
            offset += n
        y = torch.zeros_like(x, dtype=torch.float32)
        if self.ep_size > 1:
            # A rank owning no selected expert for this microbatch is routine on
            # an imbalanced route, and without this its routed_x and gate outputs
            # would have no differentiable consumer at all: autograd would never
            # reach the seam, its all-reduce would never be issued, and the peers
            # that did issue theirs would hang. The zero terms keep the graph --
            # and so the collective order -- identical on every rank, and add
            # nothing to the value. The expert parameters are attached for the
            # same reason, so DDP still sees every registered parameter.
            y = (
                y
                + routed_x.float() * 0.0
                + weights.float().sum() * 0.0
                + sum(
                    getattr(e, name).reshape(-1)[0].float() * 0.0
                    for name in e._WEIGHT_NAMES
                )
            )
        if y_parts:
            y = y.index_add(0, token_of, torch.cat(y_parts, dim=0).float())
        if self.ep_size > 1:
            y = reduce_routed_output(y, self.experts.ep_group)
        y = y + self.shared_experts(x)
        return y.to(x.dtype).view(shape)
