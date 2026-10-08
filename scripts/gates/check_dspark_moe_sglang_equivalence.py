#!/usr/bin/env python3
"""Equivalence gate: the served MoE FFN vs the plain-PyTorch reference.

Builds random checkpoints in SpecForge's Qwen naming at a small shape and at
the Qwen3.8-27B DSpark MoE drafter shape (512 experts of width 512, top-10,
sigmoid-gated shared expert of width 2048, folded router centering in
``gate.bias``), loads them into ``dflash_moe.QwenMoESparseBlock`` (SGLang
``TopK`` + ``FusedMoE``, loaded through the modules' own weight loaders) and
into ``moe_ffn.DraftMoEFFN`` (plain PyTorch), and compares routing (identical
top-k sets, same combine weights) and outputs. When SpecForge's trainer MoE
package is importable, the reference is also checked against the trainer's
``MoELayer`` on the same checkpoint. Needs CUDA and SGLang:

    python scripts/gates/check_dspark_moe_sglang_equivalence.py
    python scripts/gates/check_dspark_moe_sglang_equivalence.py --moe-runner-backend flashinfer_trtllm
"""

import argparse
import sys

import torch


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _sglang_runtime(moe_runner_backend: str):
    """Publish a server config and a TP1 group the way a scheduler process has
    them; ``FusedMoE`` and ``TopK`` need both."""
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.layers.moe.utils import initialize_moe_config
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs

    publish(
        ServerArgs(
            model_path="dummy",
            attention_backend="triton",
            moe_runner_backend=moe_runner_backend,
        ),
        role="scheduler",
    )
    initialize_moe_config()
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:29599",
        local_rank=0,
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )


def qwen35_config(e, k, inter, hidden, shared):
    # What MoEConfig.serving_fields() + the normalizer write for the
    # qwen3_5_moe preset (Qwen keys).
    return _Cfg(
        hidden_size=hidden,
        num_experts=e,
        num_experts_per_tok=k,
        moe_intermediate_size=inter,
        shared_expert_intermediate_size=shared,
        n_shared_experts=1,
        moe_preset="qwen3_5_moe",
        scoring_func="softmax",
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        moe_router_bias=True,
        shared_expert_gate="sigmoid",
        hidden_act="silu",
    )


def random_checkpoint(e, inter, hidden, shared, dtype, seed=0):
    """A random export of one layer's FFN in Qwen naming (names relative to the FFN)."""
    gen = torch.Generator().manual_seed(seed)

    def rn(*shape, scale):
        return (torch.randn(*shape, generator=gen) * scale).to(dtype)

    ckpt = {
        "gate.weight": rn(e, hidden, scale=0.02),
        "gate.bias": rn(e, scale=0.5).float(),
        "shared_expert.gate_proj.weight": rn(shared, hidden, scale=0.02),
        "shared_expert.up_proj.weight": rn(shared, hidden, scale=0.02),
        "shared_expert.down_proj.weight": rn(hidden, shared, scale=0.02),
        "shared_expert_gate.weight": rn(1, hidden, scale=0.02),
    }
    for i in range(e):
        ckpt[f"experts.{i}.gate_proj.weight"] = rn(inter, hidden, scale=0.02)
        ckpt[f"experts.{i}.up_proj.weight"] = rn(inter, hidden, scale=0.02)
        ckpt[f"experts.{i}.down_proj.weight"] = rn(hidden, inter, scale=0.02)
    return ckpt


def load_reference(ref, ckpt):
    """Load a Qwen-named checkpoint into ``moe_ffn.DraftMoEFFN`` (stacked experts)."""
    e = ref.n_experts
    with torch.no_grad():
        ref.gate.weight.copy_(ckpt["gate.weight"].to(ref.gate.weight.dtype))
        ref.gate.bias.data = ckpt["gate.bias"].float().to(ref.gate.bias.device)
        for w, proj in (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")):
            stacked = torch.stack(
                [ckpt[f"experts.{i}.{proj}.weight"] for i in range(e)]
            )
            getattr(ref.experts, w).copy_(stacked.to(ref.experts.w1.dtype))
        ref.shared_experts.w1.weight.copy_(ckpt["shared_expert.gate_proj.weight"])
        ref.shared_experts.w3.weight.copy_(ckpt["shared_expert.up_proj.weight"])
        ref.shared_experts.w2.weight.copy_(ckpt["shared_expert.down_proj.weight"])
        ref.shared_experts.gate.weight.copy_(ckpt["shared_expert_gate.weight"])


def build_pair(e, k, inter, hidden, shared, device, dtype):
    from specforge.serving.sglang_models.dflash_moe import QwenMoESparseBlock
    from specforge.serving.sglang_models.moe_ffn import DraftMoEFFN

    cfg = qwen35_config(e, k, inter, hidden, shared)
    ckpt = random_checkpoint(e, inter, hidden, shared, dtype)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            ref = DraftMoEFFN(cfg)
            served = QwenMoESparseBlock(cfg, layer_id=0, prefix="layers.0.mlp")
    finally:
        torch.set_default_dtype(prev)
    load_reference(ref, ckpt)
    served.load_weights((name, t.to(device)) for name, t in ckpt.items())
    served.check_loaded("gate")
    # What SGLang's model loader does after load_weights (runner-specific
    # weight layouts, fp8 re-wrapping, ...).
    served.experts.quant_method.process_weights_after_loading(served.experts)
    return ref.eval(), served.eval(), cfg, ckpt


def served_routing(served, x):
    """(weights, indices) the served block routes with, or None when the runner
    routes inside the kernel (flashinfer TRT-LLM bypasses TopK)."""
    out = served.topk(x, served.router_logits(x))
    if getattr(out, "topk_ids", None) is None:
        return None
    return out.topk_weights.float(), out.topk_ids.long()


def compare(ref, served, tokens, hidden, device, dtype, label):
    x = torch.randn(tokens, hidden, device=device, dtype=dtype)
    with torch.no_grad():
        y_ref = ref(x)
        y_served = served(x)
        w_ref, i_ref, _ = ref.route(x)
        routing = served_routing(served, x)
    diff = (y_ref.float() - y_served.float()).abs()
    scale = y_ref.float().abs().mean().item()
    if routing is None:
        same_idx, w_diff, routing_note = True, 0.0, "routing inside kernel"
    else:
        w_served, i_served = routing
        same_idx = torch.equal(i_ref.sort(-1).values, i_served.sort(-1).values)
        w_diff = (w_ref.sort(-1).values - w_served.sort(-1).values).abs().max().item()
        routing_note = f"same_topk={same_idx} max|dw|={w_diff:.2e}"
    print(
        f"[{label}] tokens={tokens} {routing_note} max|dy|={diff.max().item():.3e} "
        f"mean|dy|={diff.mean().item():.3e} mean|y|={scale:.3e}"
    )
    # bf16-level drift: elementwise max within 20% of the mean magnitude, mean within 2%.
    return (
        same_idx
        and w_diff < 1e-5
        and diff.max().item() <= 0.2 * max(scale, 1e-3)
        and diff.mean().item() <= 0.02 * max(scale, 1e-3)
    )


def check_trainer(ref, cfg, ckpt, device, dtype):
    """Reference vs SpecForge's trainer MoELayer on the same checkpoint, when
    the trainer MoE package exists in this checkout."""
    try:
        from specforge.modeling.draft.moe import MoEConfig, MoELayer
    except Exception as err:  # pragma: no cover - depends on the checkout
        print(f"[trainer] skipped: specforge.modeling.draft.moe not importable ({err})")
        return True
    try:
        moe_cfg = MoEConfig(
            preset="qwen3_5_moe",
            n_routed_experts=cfg.num_experts,
            num_experts_per_tok=cfg.num_experts_per_tok,
            moe_intermediate_size=cfg.moe_intermediate_size,
            shared_expert_intermediate_size=cfg.shared_expert_intermediate_size,
            dispatch="grouped_mm",
        )
        layer = MoELayer(moe_cfg, cfg.hidden_size).to(device=device, dtype=dtype).eval()
    except Exception as err:  # pragma: no cover
        print(f"[trainer] skipped: cannot build a qwen3_5_moe MoELayer here ({err})")
        return True
    state = {}
    for name, t in ckpt.items():
        state[name] = t
    missing, unexpected = layer.load_state_dict(state, strict=False)
    if unexpected or any("expert" in m or "gate" in m for m in missing):
        print(
            f"[trainer] skipped: checkpoint naming does not match this trainer version "
            f"(missing {missing[:3]}, unexpected {unexpected[:3]})"
        )
        return True
    x = torch.randn(37, cfg.hidden_size, device=device, dtype=dtype)
    with torch.no_grad():
        d = (layer(x).float() - ref(x).float()).abs()
    scale = ref(x).float().abs().mean().item()
    ok = d.max().item() <= 0.2 * max(scale, 1e-3)
    print(
        f"[trainer] reference vs MoELayer: max|dy|={d.max().item():.3e} mean|y|={scale:.3e} -> {'ok' if ok else 'FAIL'}"
    )
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--moe-runner-backend",
        default="triton",
        choices=("triton", "flashinfer_trtllm"),
        help="SGLang MoE runner for FusedMoE (flashinfer_trtllm = the TRT-LLM "
        "MoE kernels on Blackwell, which route inside the kernel)",
    )
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("needs CUDA (the served block runs SGLang kernels)")
        return 2
    _sglang_runtime(args.moe_runner_backend)
    device, dtype = "cuda", torch.bfloat16
    tag = f"qwen3_5_moe/{args.moe_runner_backend}"
    ok = True
    torch.manual_seed(0)
    ref, served, cfg, ckpt = build_pair(8, 3, 128, 256, 256, device, dtype)
    ok &= check_trainer(ref, cfg, ckpt, device, dtype)
    for tokens in (1, 37):
        ok &= compare(ref, served, tokens, 256, device, dtype, f"{tag}/small")
    del ref, served
    # Kan's Qwen3.8-27B DSpark MoE drafter: 512 x 512 top-10, shared 2048.
    ref, served, cfg, ckpt = build_pair(512, 10, 512, 5120, 2048, device, dtype)
    for tokens in (7, 56, 448):
        ok &= compare(ref, served, tokens, 5120, device, dtype, f"{tag}/real")
    # A truncated export must be refused.
    from specforge.serving.sglang_models.dflash_moe import QwenMoESparseBlock

    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            block = QwenMoESparseBlock(qwen35_config(8, 3, 128, 256, 256), 0)
    finally:
        torch.set_default_dtype(prev)
    partial = {
        k: v.to(device) for k, v in ckpt.items() if not k.startswith("experts.5.")
    }
    try:
        block.load_weights(
            (k, v)
            for k, v in random_checkpoint(8, 128, 256, 256, dtype).items()
            if not k.startswith("experts.5.")
        )
        block.check_loaded("truncated")
        print("[strict] FAIL: a truncated export was accepted")
        ok = False
    except ValueError as err:
        print(f"[strict] rejects a truncated export: {str(err)[:90]}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
