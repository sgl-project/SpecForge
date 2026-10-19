# coding=utf-8
"""DeepSeek-V4 MoE family: DeepSeek-V4-Flash(-0731) and the DSpark drafter
bundled with it.

**Preset ``deepseek_v4``**: ``sqrt(softplus)`` scores; top-k on the scores
plus an aux-loss-free selection bias (``noaux_tc``: after every step the sign
rule moves each expert's bias against its load); the combine weights are the
*unbiased* scores of the chosen experts, renormalized and scaled by 1.5; one
ungated SwiGLU shared expert as wide as a routed expert; SwiGLU clamped at 10
(``up`` on both sides, ``gate`` from above). The targets set
``n_group == topk_group``, so group-limited routing is off.

**Layout ``deepseek``**: the official checkpoint naming, which DeepSeek's
``inference/convert.py`` and SGLang's DSpark loader read::

    gate.weight                     router [E, hidden]
    gate.bias                       selection bias, fp32   <- gate.balance.bias
    experts.{i}.w1|w2|w3.weight     <- experts.w1|w2|w3 [E, out, in]
    shared_experts.w1|w2|w3.weight  (module names)

Hash-routed target layers (``gate.tid2eid``) have no learned router and are
rejected.
``gate.bias`` only steers selection here, so router centering, which an
export folds into a router *logit* bias, has no name in this layout.

**Serving**: a DSpark drafter rides inside the target checkpoint as
``mtp.*``. SGLang's ``DeepseekV4ForCausalLMDSpark`` builds its MoE from the
target's routing config and overrides only the expert counts, from
``dspark_n_routed_experts`` and ``dspark_num_experts_per_tok``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Sequence

import torch

from ..checkpoint import register_layout
from ..config import MoEConfig, register_moe_preset

register_moe_preset(
    "deepseek_v4",
    scoring_func="sqrtsoftplus",
    norm_topk_prob=True,
    routed_scaling_factor=1.5,
    n_shared_experts=1,
    swiglu_limit=10.0,
    router="topk",
    balance="noaux_tc",
    shared_expert_gate="none",
    layout="deepseek",
)


# -- layout ------------------------------------------------------------------------

_EXPERT = re.compile(r"^experts\.(\d+)\.(w[123])\.weight$")
#: Tensors with one row per routed expert besides the experts themselves.
_ROUTER_ROWS = ("gate.weight", "gate.bias")


def to_deepseek_file(layer: Dict[str, Any]) -> Dict[str, Any]:
    """One MoE layer, module naming -> DeepSeek file naming."""
    if "gate.bias" in layer:
        raise ValueError(
            "the deepseek layout's gate.bias is the selection-only bias, so it "
            "cannot hold a router logit bias (moe_router_bias, or the one "
            "moe_router_center folds into at export); train without them or "
            "export with moe_layout='native'"
        )
    out: Dict[str, Any] = {}
    for key, value in layer.items():
        if key in ("experts.w1", "experts.w2", "experts.w3"):
            name = key[len("experts.") :]
            for i in range(value.shape[0]):
                out[f"experts.{i}.{name}.weight"] = value[i]
        elif key == "gate.balance.bias":
            out["gate.bias"] = value
        else:
            out[key] = value
    return out


def from_deepseek_file(layer: Dict[str, Any]) -> Dict[str, Any]:
    """One MoE layer, DeepSeek file naming -> module naming."""
    groups: Dict[str, Dict[int, torch.Tensor]] = {}
    out: Dict[str, Any] = {}
    for key, value in layer.items():
        if m := _EXPERT.match(key):
            groups.setdefault(m[2], {})[int(m[1])] = value
        elif key == "gate.bias":
            out["gate.balance.bias"] = value
        elif key == "gate.tid2eid":
            raise ValueError(
                "hash-routed MoE layer (gate.tid2eid): its experts are picked by "
                "token id, not by a learned router; use a learned-routing layer"
            )
        else:
            out[key] = value
    for name, members in groups.items():
        n = max(members) + 1
        if sorted(members) != list(range(n)):
            raise KeyError(
                f"experts.*.{name}.weight is missing expert indices: "
                f"have {sorted(members)}"
            )
        out[f"experts.{name}"] = torch.stack([members[i] for i in range(n)], dim=0)
    return out


def select_deepseek_experts(
    layer: Dict[str, Any], ids: Sequence[int]
) -> Dict[str, Any]:
    """Keep experts ``ids`` (renumbered ``0..len(ids)-1``) with their router rows."""
    rows = torch.as_tensor(list(ids), dtype=torch.long)
    out: Dict[str, Any] = {}
    for new, old in enumerate(ids):
        for w in ("w1", "w2", "w3"):
            out[f"experts.{new}.{w}.weight"] = layer[f"experts.{old}.{w}.weight"]
    for key, value in layer.items():
        if key in _ROUTER_ROWS:
            out[key] = value.index_select(0, rows)
        elif key.startswith("shared_experts."):
            out[key] = value
    return out


#: The routing SGLang's DeepseekV4ForCausalLMDSpark takes from the target
#: config; a drafter may differ from its target only in the expert counts.
TARGET_ROUTING = {
    "scoring_func": "sqrtsoftplus",
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.5,
    "swiglu_limit": 10.0,
    "n_shared_experts": 1,
    "shared_expert_gate": "none",
    "balance": "noaux_tc",
}


def deepseek_serving_config(cfg: MoEConfig) -> Dict[str, Any]:
    """``config.json`` fields for ``DeepseekV4ForCausalLMDSpark``.

    Refuses recipes that class cannot run, so an export never silently serves
    a different routing than the one trained.
    """
    problems = [
        f"{name}={getattr(cfg, name)!r} (DeepSeek-V4 uses {value!r})"
        for name, value in TARGET_ROUTING.items()
        if getattr(cfg, name) != value
    ]
    if cfg.group_limited:
        problems.append("group-limited routing")
    if cfg.shared_expert_intermediate_size != cfg.moe_intermediate_size:
        problems.append("a shared expert wider than the routed experts")
    if cfg.router_center != "none":
        problems.append(f"moe_router_center={cfg.router_center!r}")
    if cfg.router_bias:
        problems.append("a router logit bias")
    if problems:
        raise ValueError(
            "DeepseekV4ForCausalLMDSpark cannot serve this MoE recipe: "
            + ", ".join(problems)
            + "; export with moe_layout='native' for SpecForge-only evaluation"
        )
    return {
        "dspark_n_routed_experts": cfg.n_routed_experts,
        "dspark_num_experts_per_tok": cfg.num_experts_per_tok,
    }


register_layout(
    "deepseek",
    to_file=to_deepseek_file,
    from_file=from_deepseek_file,
    serving_config=deepseek_serving_config,
)
