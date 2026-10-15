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

        if self.grouped_mm and x.is_cuda:
            offs = counts.cumsum(0).to(torch.int32)
            gate = torch._grouped_mm(x_sorted, self.w1.transpose(-1, -2), offs=offs)
            up = torch._grouped_mm(x_sorted, self.w3.transpose(-1, -2), offs=offs)
            h = w_sorted * swiglu_clamped(gate, up, self.swiglu_limit)
            y_routed = torch._grouped_mm(
                h.to(x.dtype), self.w2.transpose(-1, -2), offs=offs
            )
        else:
            counts_list = counts.tolist()  # one host sync per MoE forward
            parts = []
            offset = 0
            for i, n in enumerate(counts_list):
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
            if not parts:
                return torch.zeros_like(x)
            y_routed = torch.cat(parts, dim=0)

        y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        y = y.index_add(0, token_of, y_routed.float())
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
