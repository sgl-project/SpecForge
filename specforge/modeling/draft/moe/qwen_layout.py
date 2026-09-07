# coding=utf-8
"""Qwen MoE checkpoint naming (``Qwen2MoeSparseMoeBlock`` in SGLang / HF).

The module layout is family-neutral (``gate``, ``experts.w{1,2,3}``,
``shared_experts.w{1,2,3}`` [+ ``shared_experts.gate``]). DeepSeek-family
files keep those names per expert; Qwen-family files (``qwen3_5_moe_text``,
``qwen3_next``, ``qwen3_moe``) use::

    gate.weight                             (unchanged)
    experts.{i}.gate_proj.weight            <- experts.{i}.w1.weight
    experts.{i}.up_proj.weight              <- experts.{i}.w3.weight
    experts.{i}.down_proj.weight            <- experts.{i}.w2.weight
    shared_expert.{gate,up,down}_proj.weight <- shared_experts.w{1,3,2}.weight
    shared_expert_gate.weight  [1, hidden]  <- shared_experts.gate.weight

which SGLang's ``FusedMoE.make_expert_params_mapping(ckpt_gate_proj_name=
"gate_proj", ckpt_down_proj_name="down_proj", ckpt_up_proj_name="up_proj")``
and the ``shared_expert`` / ``shared_expert_gate`` submodules load
(``sglang/srt/models/qwen2_moe.py``, used by ``qwen3_next.py`` /
``qwen3_5_text.py``).

Direction rules: a layer is written in Qwen naming iff it carries the
per-token shared-expert gate (``shared_experts.gate.weight``), which only the
Qwen family has; reading accepts Qwen naming unconditionally (the names are
unambiguous). Both directions are idempotent and no-ops on dense models and
on DeepSeek-layout dicts.
"""

from __future__ import annotations

import re
from typing import Dict

from .state_dict import register_state_dict_converter

W_TO_QWEN = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
QWEN_TO_W = {v: k for k, v in W_TO_QWEN.items()}

_NATIVE_SHARED_GATE = re.compile(r"^(?P<base>(?:.*\.)?)shared_experts\.gate\.weight$")
_NATIVE_EXPERT = re.compile(
    r"^(?P<base>(?:.*\.)?)experts\.(?P<idx>\d+)\.(?P<w>w[123])\.weight$"
)
_NATIVE_SHARED = re.compile(
    r"^(?P<base>(?:.*\.)?)shared_experts\.(?P<w>w[123])\.weight$"
)

_QWEN_SHARED_GATE = re.compile(r"^(?P<base>(?:.*\.)?)shared_expert_gate\.weight$")
_QWEN_EXPERT = re.compile(
    r"^(?P<base>(?:.*\.)?)experts\.(?P<idx>\d+)\.(?P<p>gate_proj|up_proj|down_proj)\.weight$"
)
_QWEN_SHARED = re.compile(
    r"^(?P<base>(?:.*\.)?)shared_expert\.(?P<p>gate_proj|up_proj|down_proj)\.weight$"
)


def qwen_layout_bases(state: Dict[str, object]) -> set:
    """Layer prefixes (``"layers.0.mlp."`` style, possibly ``""``) written in
    Qwen naming: those with a per-token shared-expert gate."""
    return {m["base"] for k in state if (m := _NATIVE_SHARED_GATE.match(k))}


def to_qwen_layout(state: Dict[str, object]) -> Dict[str, object]:
    bases = qwen_layout_bases(state)
    if not bases:
        return state
    out = {}
    for key, value in state.items():
        if (m := _NATIVE_SHARED_GATE.match(key)) and m["base"] in bases:
            key = f"{m['base']}shared_expert_gate.weight"
        elif (m := _NATIVE_EXPERT.match(key)) and m["base"] in bases:
            key = f"{m['base']}experts.{m['idx']}.{W_TO_QWEN[m['w']]}.weight"
        elif (m := _NATIVE_SHARED.match(key)) and m["base"] in bases:
            key = f"{m['base']}shared_expert.{W_TO_QWEN[m['w']]}.weight"
        out[key] = value
    return out


def from_qwen_layout(state: Dict[str, object]) -> Dict[str, object]:
    out = {}
    for key, value in state.items():
        if m := _QWEN_SHARED_GATE.match(key):
            key = f"{m['base']}shared_experts.gate.weight"
        elif m := _QWEN_EXPERT.match(key):
            key = f"{m['base']}experts.{m['idx']}.{QWEN_TO_W[m['p']]}.weight"
        elif m := _QWEN_SHARED.match(key):
            key = f"{m['base']}shared_experts.{QWEN_TO_W[m['p']]}.weight"
        out[key] = value
    return out


# Registered after the grouped-experts converter so that on the way out the
# stacked tensors are already per-expert, and on the way in the Qwen names are
# turned into per-expert ``w{1,2,3}`` before they are stacked.
register_state_dict_converter(
    "qwen_layout", to_checkpoint=to_qwen_layout, from_checkpoint=from_qwen_layout
)
