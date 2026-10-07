#!/usr/bin/env python3
"""Equivalence test: SGLang's DraftMoEFFN vs SpecForge's MoELayer (Kan's code).

Builds SpecForge's MoE FFN with random weights (real DSV4-preset sizes and a small
config), converts its state to the official checkpoint naming, loads it into the
SGLang draft FFN through the same stacking logic the serving loader uses, and
compares outputs on random inputs for both dispatch paths. Needs an SGLang with
``patches/sglang/v0.5.18/dspark-moe-draft.patch`` applied (it imports
``sglang.srt.models.dspark_moe``); the grouped_mm cases need a CUDA device:

    python scripts/gates/check_dspark_moe_sglang_equivalence.py
"""
import sys

import torch
from sglang.srt.models.dspark_moe import DraftMoEFFN, stack_expert_weights

from specforge.modeling.draft.moe import MoEConfig, MoELayer, to_checkpoint_state_dict


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def build_pair(e, k, inter, hidden, device, dtype, shared=1, bias_scale=0.0):
    moe_cfg = MoEConfig(
        preset="deepseek_v4",
        n_routed_experts=e,
        num_experts_per_tok=k,
        moe_intermediate_size=inter,
        n_shared_experts=shared,
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        router="topk",
        balance="noaux_tc",
        experts_backend="grouped",
        shared_expert="swiglu",
        shared_expert_gate="none",
        dispatch="grouped_mm",
    )
    ref = MoELayer(moe_cfg, hidden)
    ref.reset_parameters(0.02)
    if bias_scale:
        # Simulate a trained selection bias so selection != raw score order.
        ref.gate.balance.bias.normal_(0, bias_scale)
    ref = ref.to(device=device, dtype=dtype).eval()
    ref.gate.balance.bias.data = ref.gate.balance.bias.data.float()

    state = to_checkpoint_state_dict(
        {k_: v.detach() for k_, v in ref.state_dict().items()}
    )
    keys = sorted(state)
    assert f"experts.0.w1.weight" in keys and "gate.bias" in keys, keys[:5]
    if shared:
        assert "shared_experts.w1.weight" in keys

    sg_cfg = _Cfg(
        hidden_size=hidden,
        n_routed_experts=e,
        num_experts_per_tok=k,
        moe_intermediate_size=inter,
        n_shared_experts=shared,
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        n_group=1,
        topk_group=1,
        topk_method="noaux_tc",
        swiglu_limit=10.0,
        hidden_act="silu",
        moe_preset="deepseek_v4",
    )
    with torch.device(device):
        sg = DraftMoEFFN(sg_cfg).to(dtype=dtype)
    sg.gate.bias.data = sg.gate.bias.data.float()
    stacked = dict(stack_expert_weights(list(state.items())))
    sg_params = dict(sg.named_parameters())
    assert set(stacked) == set(sg_params), (
        sorted(set(stacked) - set(sg_params)),
        sorted(set(sg_params) - set(stacked)),
    )
    with torch.no_grad():
        for name, tensor in stacked.items():
            sg_params[name].copy_(tensor.to(sg_params[name].dtype))
    return ref.eval(), sg.eval()


def compare(ref, sg, tokens, hidden, device, dtype, label):
    x = torch.randn(tokens, hidden, device=device, dtype=dtype)
    with torch.no_grad():
        y_ref = ref(x)
        y_sg = sg(x)
        # Routing must be identical, not just close.
        w_ref, i_ref = ref.gate(x).weights, ref.gate(x).indices
        w_sg, i_sg, _ = sg.route(x)
    same_idx = torch.equal(i_ref.sort(-1).values, i_sg.sort(-1).values)
    w_diff = (w_ref.sort(-1).values - w_sg.sort(-1).values).abs().max().item()
    diff = (y_ref.float() - y_sg.float()).abs()
    scale = y_ref.float().abs().mean().item()
    print(
        f"[{label}] tokens={tokens} same_topk={same_idx} max|dw|={w_diff:.2e} "
        f"max|dy|={diff.max().item():.3e} mean|dy|={diff.mean().item():.3e} "
        f"mean|y|={scale:.3e}"
    )
    ok = (
        same_idx and w_diff < 1e-5 and diff.max().item() <= 2e-2 * max(scale, 1e-3) * 10
    )
    return ok


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    ok = True
    torch.manual_seed(0)
    # Small config, with a non-trivial selection bias.
    ref, sg = build_pair(8, 2, 64, 128, device, dtype, shared=1, bias_scale=0.3)
    ok &= compare(ref, sg, 1, 128, device, dtype, "small/grouped_mm")
    ok &= compare(ref, sg, 37, 128, device, dtype, "small/grouped_mm")
    # Loop path (force by moving to CPU) vs reference on CPU.
    ref_cpu, sg_cpu = ref.to("cpu"), sg.to("cpu")
    ref_cpu.experts.grouped_mm = False
    ok &= compare(ref_cpu, sg_cpu, 23, 128, "cpu", dtype, "small/loop-cpu")
    # Real DSV4-preset sizes.
    ref, sg = build_pair(64, 6, 2048, 4096, device, dtype, shared=1, bias_scale=1.0)
    ok &= compare(ref, sg, 8, 4096, device, dtype, "dsv4/grouped_mm")
    ok &= compare(ref, sg, 256, 4096, device, dtype, "dsv4/grouped_mm")
    # Stacking must reject a truncated export.
    try:
        stack_expert_weights(
            [
                ("layers.0.mlp.experts.0.w1.weight", torch.zeros(2, 2)),
                ("layers.0.mlp.experts.2.w1.weight", torch.zeros(2, 2)),
            ]
        )
        print("[stack] FAIL: missing expert index not rejected")
        ok = False
    except ValueError as err:
        print(f"[stack] rejects gaps: {str(err)[:80]}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
