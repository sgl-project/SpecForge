# coding=utf-8
"""Target-family MoE presets.

A preset is the routing recipe of one target family; the draft JSON adds the
per-run sizes (``n_routed_experts``, ``num_experts_per_tok``,
``moe_intermediate_size``) and may override any key for ablations.
"""

from .config import register_moe_preset

# DeepSeek-V4 (e.g. DeepSeek-V4-Flash): sqrt(softplus) scores, aux-loss-free
# top-k with the sign-controlled selection bias, renormalized combine weights
# scaled by 1.5, one ungated shared expert, SwiGLU clamped at 10. The target's
# n_group == topk_group, so group-limited routing is off by default.
register_moe_preset(
    "deepseek_v4",
    scoring_func="sqrtsoftplus",
    norm_topk_prob=True,
    routed_scaling_factor=1.5,
    n_shared_experts=1,
    swiglu_limit=10.0,
    router="topk",
    balance="noaux_tc",
    experts_backend="grouped",
    shared_expert="swiglu",
    shared_expert_gate="none",
)

# Qwen3.5/3.8 MoE (``qwen3_5_moe_text``, e.g. Qwen3.8-2.4T-A95B): softmax
# scores, plain top-k with the top-k probabilities renormalized to sum 1 and
# no scaling, no selection bias; balancing is the Switch-Transformer auxiliary
# load-balancing loss (transformers' ``load_balancing_loss_func``). One SwiGLU
# shared expert of ``shared_expert_intermediate_size`` width, gated per token
# by ``sigmoid(shared_expert_gate(x))``; no SwiGLU clamp. Checkpoints use the
# Qwen naming SGLang's ``Qwen2MoeSparseMoeBlock`` loads (see qwen_layout.py).
register_moe_preset(
    "qwen3_5_moe",
    scoring_func="softmax",
    norm_topk_prob=True,
    routed_scaling_factor=1.0,
    n_shared_experts=1,
    swiglu_limit=0.0,
    router="topk",
    balance="aux_loss",
    experts_backend="grouped",
    shared_expert="swiglu",
    shared_expert_gate="sigmoid",
)
