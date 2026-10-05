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

# Qwen3-MoE (e.g. Qwen3-30B-A3B, Qwen3-235B-A22B): softmax scores over all
# experts, top-k combine weights renormalized (``norm_topk_prob``), no scale,
# no shared expert, plain SwiGLU. Qwen3 balances with a router auxiliary loss
# in pretraining; the drafter defaults to the aux-loss-free selection bias and
# can add the complementary balance term with ``dflash_config.moe_aux_loss_coeff``.
# Qwen checkpoints name the expert count ``num_experts``; the draft JSON may
# use that name or DeepSeek's ``n_routed_experts``.
register_moe_preset(
    "qwen3",
    scoring_func="softmax",
    norm_topk_prob=True,
    routed_scaling_factor=1.0,
    n_shared_experts=0,
    swiglu_limit=0.0,
    router="topk",
    balance="noaux_tc",
    experts_backend="grouped",
    shared_expert="swiglu",
    shared_expert_gate="none",
)
