# coding=utf-8
"""Qwen3.5 / Qwen3.8 MoE family: the ``qwen3_5_moe`` preset, the ``qwen`` layout, serving.

File naming, what SGLang's ``Qwen2MoeSparseMoeBlock`` loader reads::

    gate.weight                                      router [E, hidden]
    gate.bias                                        router logit bias (centering exports)
    experts.{i}.gate_proj|up_proj|down_proj.weight   <- experts.w1|w3|w2 [E, out, in]
    shared_expert.gate_proj|up_proj|down_proj.weight <- shared_experts.w1|w3|w2.weight
    shared_expert_gate.weight [1, hidden]            <- shared_experts.gate.weight

Serving: SGLang's ``Qwen3MoeDSparkModel`` (not upstream yet) reads the fields below.
"""

from __future__ import annotations

import re
from typing import Any, Dict

import torch

from ..checkpoint import register_layout
from ..config import MoEConfig, register_moe_preset

register_moe_preset(
    "qwen3_5_moe",
    scoring_func="softmax",
    norm_topk_prob=True,
    routed_scaling_factor=1.0,
    n_shared_experts=1,
    router="topk",
    balance="none",
    shared_expert_gate="sigmoid",
    layout="qwen",
)

W_TO_QWEN = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
QWEN_TO_W = {v: k for k, v in W_TO_QWEN.items()}

_QWEN_EXPERT = re.compile(r"^experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
_QWEN_SHARED = re.compile(r"^shared_expert\.(gate_proj|up_proj|down_proj)\.weight$")


def to_qwen_file(layer: Dict[str, Any]) -> Dict[str, Any]:
    """One MoE layer, module naming -> Qwen file naming."""
    out: Dict[str, Any] = {}
    for key, value in layer.items():
        name = key.split(".")
        if key in ("experts.w1", "experts.w2", "experts.w3"):
            proj = W_TO_QWEN[name[1]]
            for i in range(value.shape[0]):
                out[f"experts.{i}.{proj}.weight"] = value[i]
        elif key.startswith("shared_experts.w") and key.endswith(".weight"):
            out[f"shared_expert.{W_TO_QWEN[name[1]]}.weight"] = value
        elif key == "shared_experts.gate.weight":
            out["shared_expert_gate.weight"] = value
        elif key == "gate.balance.bias":
            raise ValueError(
                "the qwen layout has no file name for a balance controller's "
                "selection bias; choose another moe_layout"
            )
        else:
            out[key] = value
    return out


def from_qwen_file(layer: Dict[str, Any]) -> Dict[str, Any]:
    """One MoE layer, Qwen file naming -> module naming."""
    groups: Dict[str, Dict[int, torch.Tensor]] = {}
    out: Dict[str, Any] = {}
    for key, value in layer.items():
        if m := _QWEN_EXPERT.match(key):
            groups.setdefault(QWEN_TO_W[m[2]], {})[int(m[1])] = value
        elif m := _QWEN_SHARED.match(key):
            out[f"shared_experts.{QWEN_TO_W[m[1]]}.weight"] = value
        elif key == "shared_expert_gate.weight":
            out["shared_experts.gate.weight"] = value
        else:
            out[key] = value
    for w, members in groups.items():
        n = max(members) + 1
        if sorted(members) != list(range(n)):
            raise KeyError(
                f"experts.*.{W_TO_QWEN[w]}.weight is missing expert indices: "
                f"have {sorted(members)}"
            )
        out[f"experts.{w}"] = torch.stack([members[i] for i in range(n)], dim=0)
    return out


def qwen_serving_config(cfg: MoEConfig) -> Dict[str, Any]:
    """``config.json`` for ``Qwen3MoeDSparkModel``; refuses recipes it cannot run."""
    problems = []
    if cfg.scoring_func != "softmax":
        problems.append(f"scoring_func={cfg.scoring_func!r} (softmax only)")
    if cfg.balance != "none":
        problems.append(f"balance={cfg.balance!r} (none only)")
    if cfg.n_shared_experts and cfg.shared_expert_gate != "sigmoid":
        problems.append("an ungated shared expert")
    if problems:
        raise ValueError(
            "Qwen3MoeDSparkModel cannot serve this MoE recipe: "
            + ", ".join(problems)
            + "; export with moe_layout='native' for SpecForge-only evaluation"
        )
    return {
        "n_routed_experts": cfg.n_routed_experts,
        "num_experts": cfg.n_routed_experts,
        "num_experts_per_tok": cfg.num_experts_per_tok,
        "moe_intermediate_size": cfg.moe_intermediate_size,
        "n_shared_experts": cfg.n_shared_experts,
        "shared_expert_intermediate_size": cfg.shared_expert_intermediate_size,
        "scoring_func": cfg.scoring_func,
        "norm_topk_prob": cfg.norm_topk_prob,
        "routed_scaling_factor": cfg.routed_scaling_factor,
        # A centered or biased router exports a logit bias the serving model loads.
        "moe_router_bias": cfg.router_bias or cfg.router_center != "none",
    }


register_layout(
    "qwen",
    to_file=to_qwen_file,
    from_file=from_qwen_file,
    serving_config=qwen_serving_config,
    serving_architectures={"Qwen3DSparkModel": "Qwen3MoeDSparkModel"},
)
