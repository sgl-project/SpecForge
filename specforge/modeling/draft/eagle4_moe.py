# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""EAGLE-4 MoE draft MLPs in the weight layout the serving loader reads."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

LATENT_MOE = "nemotron_latent_relu2"


def relu2(x: torch.Tensor) -> torch.Tensor:
    """Squared ReLU; older transformers builds omit this from ``ACT2FN``."""

    return F.relu(x).square()


def resolve_act_fn(name: str):
    """Look up ``hidden_act``, with a ``relu2`` fallback for transformers 4.x."""

    try:
        return ACT2FN[name]
    except KeyError:
        if name == "relu2":
            return relu2
        raise


class Eagle4Router(nn.Module):
    """Sigmoid router whose expert-selection bias balances load without a loss."""

    def __init__(self, config):
        super().__init__()
        n = config.n_routed_experts
        self.weight = nn.Parameter(
            nn.init.normal_(
                torch.empty(n, config.hidden_size), std=config.initializer_range
            )
        )
        self.register_buffer("e_score_correction_bias", torch.zeros(n))
        self.register_buffer(
            "load", torch.zeros(n, dtype=torch.long), persistent=False
        )
        self.update_rate = getattr(
            config,
            "expert_bias_update_rate",
            getattr(config, "moe_router_bias_update_rate", 1e-3),
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self.e_score_correction_bias = self.e_score_correction_bias.float()
        return self

    @torch.no_grad()
    def update_bias(self) -> float:
        """Called once per optimizer step; returns max / mean expert load."""

        load = self.load.float()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(load)
        self.load.zero_()
        mean = load.mean().clamp_min(1e-8)
        self.e_score_correction_bias += self.update_rate * (mean - load).sign()
        return (load.max() / mean).item()


class SharedExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size, act_fn, gated):
        super().__init__()
        if gated:
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = act_fn

    def forward(self, x):
        h = self.up_proj(x)
        if hasattr(self, "gate_proj"):
            h = self.act_fn(self.gate_proj(x)) * h
        else:
            h = self.act_fn(h)
        return self.down_proj(h)


class Eagle4MoE(nn.Module):
    """``sparse_swiglu`` (default) or ``nemotron_latent_relu2`` replacing the dense MLP."""

    def __init__(self, config):
        super().__init__()
        self.latent = getattr(config, "moe_architecture", "sparse_swiglu") == LATENT_MOE
        self.top_k = config.num_experts_per_tok
        self.n_group = getattr(config, "n_group", None) or 1
        self.topk_group = getattr(config, "topk_group", None) or self.n_group
        self.norm_topk_prob = getattr(config, "norm_topk_prob", True)
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.act_fn = resolve_act_fn(config.hidden_act)
        n = config.n_routed_experts
        hidden = config.hidden_size
        inter = config.moe_intermediate_size
        width = config.moe_latent_size if self.latent else hidden

        self.gate = Eagle4Router(config)
        up = torch.empty(n, inter if self.latent else 2 * inter, width)
        up = nn.Parameter(nn.init.normal_(up, std=config.initializer_range))
        down = nn.Parameter(torch.zeros(n, width, inter))
        self.expert_names = (
            ("up_proj", "down_proj") if self.latent else ("w13_weight", "w2_weight")
        )
        self.experts = nn.ParameterDict(dict(zip(self.expert_names, (up, down))))
        if self.latent:
            self.fc1_latent_proj = nn.Linear(hidden, width, bias=False)
            self.fc2_latent_proj = nn.Linear(width, hidden, bias=False)
        n_shared = getattr(config, "n_shared_experts", None) or 0
        shared_inter = (
            config.moe_shared_expert_intermediate_size if self.latent else inter
        )
        self.shared_experts = (
            SharedExpert(
                hidden, shared_inter * n_shared, self.act_fn, gated=not self.latent
            )
            if n_shared
            else None
        )

    def route(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.gate.weight.float()).sigmoid()
        biased = scores + self.gate.e_score_correction_bias
        if self.n_group > 1:
            groups = biased.unflatten(-1, (self.n_group, -1))
            best = groups.topk(2, dim=-1).values.sum(-1).topk(self.topk_group, dim=-1).indices
            dropped = torch.ones_like(groups[..., 0], dtype=torch.bool).scatter_(
                1, best, False
            )
            biased = groups.masked_fill(dropped[..., None], float("-inf")).flatten(1)
        idx = biased.topk(self.top_k, dim=-1).indices
        weights = scores.gather(1, idx)
        if self.norm_topk_prob:
            weights = weights / weights.sum(-1, keepdim=True)
        return idx, weights * self.routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        idx, weights = self.route(x)
        counts = torch.bincount(idx.flatten(), minlength=self.gate.weight.shape[0])
        if self.training:
            self.gate.load += counts.to(device=self.gate.load.device)
        h = self.fc1_latent_proj(x) if self.latent else x
        up, down = (self.experts[name] for name in self.expert_names)
        out = _expert_mlp(h, idx, weights, up, down, self.latent, self.act_fn)
        if self.latent:
            out = self.fc2_latent_proj(out)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out.view_as(hidden_states)


def _grouped_mm_fn(h: torch.Tensor):
    """``F.grouped_mm`` when it is a real CUDA kernel; HIP reports ``is_cuda``."""

    grouped_mm = getattr(F, "grouped_mm", None)
    if grouped_mm is None:
        return None
    if getattr(torch.version, "hip", None):
        return None
    if not (h.is_cuda and h.device.type == "cuda"):
        return None
    return grouped_mm


def _expert_mlp(
    h: torch.Tensor,
    idx: torch.Tensor,
    weights: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    latent: bool,
    act_fn,
) -> torch.Tensor:
    grouped_mm = _grouped_mm_fn(h)
    if grouped_mm is None:
        return _expert_mlp_loop(h, idx, weights, up, down, latent, act_fn)
    try:
        return _expert_mlp_grouped(h, idx, weights, up, down, latent, act_fn, grouped_mm)
    except RuntimeError:
        return _expert_mlp_loop(h, idx, weights, up, down, latent, act_fn)


def _expert_mlp_grouped(
    h, idx, weights, up, down, latent, act_fn, grouped_mm
) -> torch.Tensor:
    order = idx.flatten().argsort()
    tokens = order // idx.shape[-1]
    counts = torch.bincount(idx.flatten(), minlength=up.shape[0])
    offs = counts.cumsum(0).int()
    inner = grouped_mm(h[tokens], up.transpose(1, 2), offs=offs)
    if latent:
        inner = act_fn(inner)
    else:
        gate, value = inner.chunk(2, dim=-1)
        inner = act_fn(gate) * value
    out = grouped_mm(inner, down.transpose(1, 2), offs=offs)
    out = out * weights.flatten()[order, None].to(out.dtype)
    return h.new_zeros(h.shape).index_add_(0, tokens, out)


def _expert_mlp_loop(
    h, idx, weights, up, down, latent, act_fn
) -> torch.Tensor:
    out = h.new_zeros(h.shape)
    n_experts = up.shape[0]
    for expert in range(n_experts):
        token_ids, slots = (idx == expert).nonzero(as_tuple=True)
        if token_ids.numel() == 0:
            continue
        hidden = h[token_ids]
        inner = F.linear(hidden, up[expert])
        if latent:
            inner = act_fn(inner)
        else:
            gate, value = inner.chunk(2, dim=-1)
            inner = act_fn(gate) * value
        contrib = F.linear(inner, down[expert])
        scale = weights[token_ids, slots].to(contrib.dtype).unsqueeze(-1)
        out.index_add_(0, token_ids, contrib * scale)
    return out


def update_eagle4_router_bias(module: nn.Module) -> Optional[float]:
    """Update every ``Eagle4Router`` under ``module``. Returns the last imbalance."""

    imbalance = None
    for child in module.modules():
        if isinstance(child, Eagle4Router):
            imbalance = child.update_bias()
    return imbalance


__all__ = [
    "Eagle4MoE",
    "Eagle4Router",
    "LATENT_MOE",
    "relu2",
    "resolve_act_fn",
    "update_eagle4_router_bias",
]
