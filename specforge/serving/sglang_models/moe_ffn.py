# coding=utf-8
"""Plain-PyTorch reference of the MoE FFN in a DFlash-family draft layer.

``DraftMoEFFN`` mirrors ``specforge.modeling.draft.moe.MoELayer`` (router
with score function, optional selection or logit bias, group limit,
renormalisation, scaling; routed experts as stacked ``w1/w2/w3`` through
``torch._grouped_mm``; optional sigmoid-gated shared expert) with no SGLang
import, so it runs on CPU. It is the reference the equivalence gate
(``scripts/gates/check_dspark_moe_sglang_equivalence.py``) and the unit tests
compare the served FFN (``dflash_moe.QwenMoESparseBlock``, built from SGLang's
``TopK`` + ``FusedMoE``) against; it is not used for serving.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

SCORE_FUNCTIONS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "softmax": lambda logits: logits.softmax(dim=-1),
    "sigmoid": torch.sigmoid,
    # DeepSeek-V4 scoring.
    "sqrtsoftplus": lambda logits: F.softplus(logits).sqrt(),
}

#: Preset defaults, used only for keys the export config does not state
#: (``MoEConfig.serving_fields()`` normally writes all of them).
PRESET_DEFAULTS: Dict[str, Dict[str, object]] = {
    "deepseek_v4": dict(
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        n_shared_experts=1,
        swiglu_limit=10.0,
        topk_method="noaux_tc",
    ),
    "qwen3": dict(
        scoring_func="softmax",
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        n_shared_experts=0,
        swiglu_limit=0.0,
        topk_method="noaux_tc",
    ),
    # Qwen3.5/3.8-MoE style (SpecForge ``qwen3_5_moe``): softmax top-k with
    # renormalisation, no selection bias, one sigmoid-gated shared expert; the
    # export folds router input centering into a pre-softmax ``gate.bias``
    # (``moe_router_bias``).
    "qwen3_5_moe": dict(
        scoring_func="softmax",
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        n_shared_experts=1,
        swiglu_limit=0.0,
        topk_method="greedy",
        shared_expert_gate="sigmoid",
    ),
}


def _cfg(config, key: str, default):
    value = getattr(config, key, None)
    return default if value is None else value


def routed_expert_count(config) -> int:
    """``n_routed_experts`` (DeepSeek) or ``num_experts`` (Qwen) of a config; 0 = dense."""
    for key in ("n_routed_experts", "num_experts"):
        value = _cfg(config, key, 0)
        if value:
            return int(value)
    return 0


def swiglu_clamped(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """SwiGLU in fp32 with the DeepSeek-V4 activation clamp (``limit`` 0 = off)."""
    gate = gate.float()
    up = up.float()
    if limit > 0:
        up = torch.clamp(up, min=-limit, max=limit)
        gate = torch.clamp(gate, max=limit)
    return F.silu(gate) * up


def group_limited_mask(
    selection: torch.Tensor, n_group: int, topk_group: int
) -> torch.Tensor:
    tokens, n_experts = selection.shape
    grouped = selection.view(tokens, n_group, n_experts // n_group)
    group_scores = grouped.topk(min(2, grouped.shape[-1]), dim=-1).values.sum(-1)
    keep = group_scores.topk(topk_group, dim=-1).indices
    mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, keep, True)
    return grouped.masked_fill(~mask.unsqueeze(-1), float("-inf")).view(
        tokens, n_experts
    )


class DraftMoEGate(nn.Module):
    """Router projection plus an optional ``gate.bias``.

    ``bias_mode`` is ``"selection"`` for the DeepSeek aux-loss-free bias (added
    to the scores for the top-k choice only) or ``"logit"`` for a bias on the
    router logits before the score function (SpecForge's folded router input
    centering, ``moe_router_bias``), which also shapes the combine weights.
    """

    def __init__(
        self, n_experts: int, hidden_size: int, bias_mode: Optional[str]
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_experts, hidden_size))
        self.bias_mode = bias_mode
        if bias_mode is not None:
            # Kept fp32 like the trainer's buffers.
            self.bias = nn.Parameter(
                torch.zeros(n_experts, dtype=torch.float32), requires_grad=False
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = F.linear(x.float(), self.weight.float())
        if self.bias_mode == "logit":
            logits = logits + self.bias.float()
        return logits


class DraftRoutedExperts(nn.Module):
    """All routed experts as three stacked parameters (``w1``/``w3`` gate/up
    ``[E, inter, hidden]``, ``w2`` down ``[E, hidden, inter]``)."""

    def __init__(self, n_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        e, d, i = n_experts, hidden_size, intermediate_size
        self.w1 = nn.Parameter(torch.empty(e, i, d))
        self.w2 = nn.Parameter(torch.empty(e, d, i))
        self.w3 = nn.Parameter(torch.empty(e, i, d))


class DraftSharedExpert(nn.Module):
    """SwiGLU shared expert (``shared_experts.w{1,2,3}``), optionally gated per
    token by ``sigmoid(shared_experts.gate(x))`` (Qwen ``shared_expert_gate``)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: float,
        gated: bool = False,
    ):
        super().__init__()
        self.swiglu_limit = float(swiglu_limit)
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate = nn.Linear(hidden_size, 1, bias=False) if gated else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = swiglu_clamped(self.w1(x), self.w3(x), self.swiglu_limit)
        y = self.w2(h.to(x.dtype))
        if self.gate is not None:
            y = torch.sigmoid(self.gate(x).float()).to(y.dtype) * y
        return y


class DraftMoEFFN(nn.Module):
    """``y = experts(x, gate(x)) + shared_experts(x)`` for one draft layer."""

    def __init__(self, config) -> None:
        super().__init__()
        preset = PRESET_DEFAULTS.get(str(_cfg(config, "moe_preset", "")), {})

        def get(key, default):
            return _cfg(config, key, preset.get(key, default))

        self.hidden_size = int(config.hidden_size)
        self.n_experts = routed_expert_count(config)
        self.topk = int(_cfg(config, "num_experts_per_tok", 0))
        self.intermediate_size = int(_cfg(config, "moe_intermediate_size", 0))
        if self.n_experts <= 0 or self.topk <= 0 or self.intermediate_size <= 0:
            raise ValueError(
                "MoE draft config needs n_routed_experts (or num_experts), "
                "num_experts_per_tok and moe_intermediate_size; got "
                f"{self.n_experts}, {self.topk}, {self.intermediate_size}."
            )
        if self.topk > self.n_experts:
            raise ValueError("num_experts_per_tok must not exceed n_routed_experts")
        if _cfg(config, "hidden_act", "silu") != "silu":
            raise ValueError("MoE draft experts support only silu (SwiGLU)")

        self.scoring_func = str(get("scoring_func", "softmax"))
        if self.scoring_func not in SCORE_FUNCTIONS:
            raise ValueError(
                f"unknown scoring_func {self.scoring_func!r}; "
                f"known: {sorted(SCORE_FUNCTIONS)}"
            )
        self.score_fn = SCORE_FUNCTIONS[self.scoring_func]
        self.norm_topk_prob = bool(get("norm_topk_prob", True))
        self.routed_scaling_factor = float(get("routed_scaling_factor", 1.0))
        self.n_group = int(get("n_group", 1))
        self.topk_group = int(get("topk_group", 1))
        if self.n_group <= 0 or self.n_experts % self.n_group:
            raise ValueError("n_group must divide n_routed_experts")
        if not 0 < self.topk_group <= self.n_group:
            raise ValueError("topk_group must be in [1, n_group]")
        self.swiglu_limit = float(get("swiglu_limit", 0.0))
        if str(get("topk_method", "greedy")) == "noaux_tc":
            bias_mode: Optional[str] = "selection"
        elif bool(get("moe_router_bias", False)):
            bias_mode = "logit"
        else:
            bias_mode = None

        self.gate = DraftMoEGate(self.n_experts, self.hidden_size, bias_mode)
        # ``experts.w{1,2,3}`` stacked [E, out, in]; the loader stacks the
        # per-expert checkpoint tensors ``experts.{i}.w{1,2,3}.weight``.
        self.experts = DraftRoutedExperts(
            self.n_experts, self.hidden_size, self.intermediate_size
        )

        n_shared = int(get("n_shared_experts", 0) or 0)
        if n_shared not in (0, 1):
            raise ValueError("n_shared_experts must be 0 or 1 for MoE drafts")
        shared_width = int(
            _cfg(config, "shared_expert_intermediate_size", self.intermediate_size)
        )
        self.shared_expert_gate = str(get("shared_expert_gate", "none"))
        if self.shared_expert_gate not in ("none", "sigmoid"):
            raise ValueError(
                f"unknown shared_expert_gate {self.shared_expert_gate!r}; "
                "known: none, sigmoid"
            )
        self.shared_experts: Optional[DraftSharedExpert] = (
            DraftSharedExpert(
                self.hidden_size,
                shared_width,
                self.swiglu_limit,
                gated=self.shared_expert_gate == "sigmoid",
            )
            if n_shared
            else None
        )
        self.grouped_mm = hasattr(torch, "_grouped_mm")
        if not self.grouped_mm:
            logger.warning(
                "torch._grouped_mm is unavailable; MoE draft experts fall back to a "
                "per-expert loop with a host sync (not CUDA-graph safe)."
            )

    def describe(self) -> str:
        return (
            f"{self.n_experts} routed experts, top-{self.topk}, width "
            f"{self.intermediate_size}, shared={self.shared_experts is not None}"
            f"{'(sigmoid-gated)' if self.shared_expert_gate == 'sigmoid' else ''}, "
            f"scoring={self.scoring_func}, bias={self.gate.bias_mode}, "
            f"renorm={self.norm_topk_prob}, scale={self.routed_scaling_factor:.2f}, "
            f"swiglu_limit={self.swiglu_limit:.1f}"
        )

    def route(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (weights [T,k] fp32, indices [T,k] long, counts [E] long)."""
        scores = self.score_fn(self.gate(x))
        selection = (
            scores + self.gate.bias.float()
            if self.gate.bias_mode == "selection"
            else scores
        )
        if self.topk_group < self.n_group:
            selection = group_limited_mask(selection, self.n_group, self.topk_group)
        indices = selection.topk(self.topk, dim=-1).indices
        weights = scores.gather(1, indices)
        if self.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.routed_scaling_factor
        flat = indices.flatten()
        counts = torch.zeros(
            self.n_experts, dtype=torch.long, device=x.device
        ).scatter_add_(0, flat, torch.ones_like(flat))
        return weights, indices, counts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.hidden_size)
        weights, indices, counts = self.route(x)
        flat_expert = indices.flatten()
        order = flat_expert.argsort(stable=True)
        token_of = order // self.topk
        x_sorted = x.index_select(0, token_of)
        w_sorted = weights.reshape(-1, 1).index_select(0, order).float()
        w1, w2, w3 = self.experts.w1, self.experts.w2, self.experts.w3
        if x.is_cuda and self.grouped_mm:
            offs = counts.cumsum(0).to(torch.int32)
            gate = torch._grouped_mm(x_sorted, w1.transpose(-1, -2), offs=offs)
            up = torch._grouped_mm(x_sorted, w3.transpose(-1, -2), offs=offs)
            h = w_sorted * swiglu_clamped(gate, up, self.swiglu_limit)
            y_routed = torch._grouped_mm(h.to(x.dtype), w2.transpose(-1, -2), offs=offs)
        else:
            counts_list = counts.tolist()
            parts = []
            offset = 0
            for i, n in enumerate(counts_list):
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
            y_routed = (
                torch.cat(parts, dim=0)
                if parts
                else torch.zeros(0, self.hidden_size, dtype=x.dtype, device=x.device)
            )
        y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        y = y.index_add(0, token_of, y_routed.float()).to(x.dtype)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y.view(shape)


#: DeepSeek-style checkpoint names -> the Qwen names the served block uses.
_DEEPSEEK_TO_QWEN = (
    (re.compile(r"^shared_experts\.gate\.weight$"), "shared_expert_gate.weight"),
    (re.compile(r"^shared_experts\.w1\."), "shared_expert.gate_proj."),
    (re.compile(r"^shared_experts\.w3\."), "shared_expert.up_proj."),
    (re.compile(r"^shared_experts\.w2\."), "shared_expert.down_proj."),
    (re.compile(r"^(experts\.\d+)\.w1\."), r"\1.gate_proj."),
    (re.compile(r"^(experts\.\d+)\.w3\."), r"\1.up_proj."),
    (re.compile(r"^(experts\.\d+)\.w2\."), r"\1.down_proj."),
)


def to_qwen_names(name: str) -> str:
    """Rename one FFN-relative checkpoint entry from DeepSeek naming
    (``experts.{i}.w{1,2,3}``, ``shared_experts.*``) to the Qwen naming the
    served block's parameters use; Qwen names pass through unchanged."""
    for pattern, repl in _DEEPSEEK_TO_QWEN:
        new = pattern.sub(repl, name)
        if new != name:
            return new
    return name
