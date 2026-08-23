#!/usr/bin/env python3
"""Gate 0: parity of ``DSparkV4DraftModel`` vs the official reference impl.

Loads the dequantized official ``mtp.*`` weights into BOTH:
  1. SpecForge's training-side ``DSparkV4DraftModel`` (parallel-block forward),
  2. the reference ``inference/model.py`` ``DSparkBlock`` stack (decode-style
     forward with a ring KV cache), with its tilelang kernels shimmed in torch,
then runs one draft block through each on identical inputs and compares the
pre-norm (confidence) and post-norm hidden states position by position.

Run on CPU; no GPU or sglang required:
    python3 scripts/gates/dspark_v4_parity_check.py [--context-len 200] [--tiny]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from dspark_v4_official_weights import (  # noqa: E402
    DEFAULT_SNAPSHOT,
    load_official_mtp_state,
)

# Reference implementation CODE (architecture only, no weights) ships with
# the standalone DSpark repo; the weights compared come from DEFAULT_SNAPSHOT
# (the 0731 bundle).
REFERENCE_INFERENCE_DIR = (
    "/cluster-storage/models/models--deepseek-ai--DeepSeek-V4-Flash-DSpark/"
    "snapshots/913f0657a874f76844e2e91cbe706dbcaceeb6d7/inference"
)


# ---------------------------------------------------------------------------
# torch shims for the reference tilelang kernels
# ---------------------------------------------------------------------------

def _shim_act_quant(x, block_size=128, scale_fmt=None, scale_dtype=None,
                    inplace=False):
    """ue8m0 fused quant+dequant (serving uses ue8m0 power-of-2 scales)."""
    assert inplace, "parity shim only supports the inplace (QAT) form"
    orig = x.dtype
    xf = x.float()
    if xf.shape[-1] % block_size != 0:
        block_size = xf.shape[-1]
    grouped = xf.unflatten(-1, (-1, block_size))
    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (grouped / scale).clamp(-448.0, 448.0)
    q = q.to(torch.float8_e4m3fn).float() * scale
    x.copy_(q.flatten(-2).to(orig))
    return x


def _shim_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Dense-torch replica of the reference sparse_attn kernel semantics."""
    b, s, h, d = q.shape
    idx = topk_idxs.long()  # [b, s, topk]; -1 => masked
    valid = idx >= 0
    gathered = kv[
        torch.arange(b).view(b, 1, 1), idx.clamp(min=0)
    ]  # [b, s, topk, d]
    scores = torch.einsum("bshd,bskd->bhsk", q.float(), gathered.float())
    scores = scores * softmax_scale
    scores = scores.masked_fill(
        ~valid.unsqueeze(1), torch.finfo(torch.float32).min
    )
    sink = attn_sink.float().view(1, h, 1, 1).expand(b, -1, s, 1)
    probs = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
    out = torch.einsum("bhsk,bskd->bshd", probs, gathered.float())
    return out.to(q.dtype)


def _shim_hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4,
                            sinkhorn_iters=20, eps=1e-6):
    from specforge.modeling.draft.dspark_v4 import _hc_split_sinkhorn

    return _hc_split_sinkhorn(
        mixes.float(), hc_scale, hc_base, hc_mult, sinkhorn_iters, eps
    )


def load_reference_module():
    kernel = types.ModuleType("kernel")
    kernel.act_quant = _shim_act_quant
    kernel.fp4_act_quant = lambda *a, **k: (_ for _ in ()).throw(
        NotImplementedError("fp4_act_quant not needed for DSpark parity")
    )
    kernel.fp8_gemm = None
    kernel.fp4_gemm = None
    kernel.sparse_attn = _shim_sparse_attn
    kernel.hc_split_sinkhorn = _shim_hc_split_sinkhorn
    sys.modules["kernel"] = kernel

    path = os.path.join(REFERENCE_INFERENCE_DIR, "model.py")
    spec = importlib.util.spec_from_file_location("dspark_reference_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dspark_reference_model"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------


def build_configs(tiny: bool):
    from specforge.training.model_loading import load_draft_config_source

    cfg = load_draft_config_source(
        os.path.join(REPO_ROOT, "configs/deepseek-v4-flash-dspark-official.json")
    )
    if tiny:
        cfg.hidden_size = 64
        cfg.vocab_size = 512
        cfg.num_attention_heads = 4
        cfg.head_dim = 64
        cfg.qk_rope_head_dim = 16
        cfg.q_lora_rank = 32
        cfg.o_lora_rank = 32
        cfg.o_groups = 2
        cfg.sliding_window = 16
        cfg.n_routed_experts = 8
        cfg.num_experts_per_tok = 2
        cfg.moe_intermediate_size = 32
    return cfg


def build_reference_blocks(ref, cfg, n_layers: int, max_seq_len: int):
    args = ref.ModelArgs(
        max_batch_size=1,
        max_seq_len=max_seq_len,
        dtype="bf16",
        scale_fmt="ue8m0",
        scale_dtype="fp32",
        expert_dtype=None,
        vocab_size=cfg.vocab_size,
        dim=cfg.hidden_size,
        moe_inter_dim=cfg.moe_intermediate_size,
        n_layers=n_layers,
        n_hash_layers=0,
        n_mtp_layers=cfg.num_hidden_layers,
        n_heads=cfg.num_attention_heads,
        n_routed_experts=cfg.n_routed_experts,
        n_shared_experts=cfg.n_shared_experts,
        n_activated_experts=cfg.num_experts_per_tok,
        score_func=cfg.scoring_func,
        route_scale=cfg.routed_scaling_factor,
        swiglu_limit=cfg.swiglu_limit,
        q_lora_rank=cfg.q_lora_rank,
        head_dim=cfg.head_dim,
        rope_head_dim=cfg.qk_rope_head_dim,
        norm_eps=cfg.rms_norm_eps,
        o_groups=cfg.o_groups,
        o_lora_rank=cfg.o_lora_rank,
        window_size=cfg.sliding_window,
        compress_ratios=tuple([0] * (n_layers + cfg.num_hidden_layers)),
        original_seq_len=0,
        rope_theta=cfg.rope_theta,
        hc_mult=cfg.hc_mult,
        hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
        hc_eps=cfg.hc_eps,
        dspark_block_size=cfg.block_size,
        dspark_noise_token_id=128799,
        dspark_target_layer_ids=tuple(cfg.dflash_config["target_layer_ids"]),
        dspark_markov_rank=cfg.dflash_config["markov_rank"],
    )
    ref.default_dtype = torch.bfloat16
    ref.scale_fmt = "ue8m0"
    ref.scale_dtype = torch.float32
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        blocks = torch.nn.ModuleList(
            [
                ref.DSparkBlock(n_layers + stage, args)
                for stage in range(cfg.num_hidden_layers)
            ]
        )
    finally:
        torch.set_default_dtype(prev_dtype)
    return blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-len", type=int, default=200)
    parser.add_argument("--tiny", action="store_true",
                        help="random tiny config instead of official weights")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    cfg = build_configs(args.tiny)
    block = cfg.block_size
    n_features = len(cfg.dflash_config["target_layer_ids"])
    S = args.context_len
    n_target_layers = cfg.num_target_layers

    from specforge.modeling.auto import AutoDraftModel

    print("building SpecForge model ...")
    mine = AutoDraftModel.from_config(cfg, torch_dtype=torch.bfloat16)
    mine.eval()
    # The reference inference impl computes the weightless q rms-scale fully
    # in bf16; sglang serving (the real target) uses fp32. Flip the model to
    # reference math for this comparison so the gate isolates structural bugs.
    for stage in mine.mtp:
        stage.attn.qscale_ref_compat = True

    if args.tiny:
        state = {
            k: (torch.randn_like(v) * 0.02 if v.is_floating_point() else v)
            for k, v in mine.state_dict().items()
        }
        # keep the sinkhorn inputs in a realistic range
        for key, value in state.items():
            if "hc_" in key and key.endswith("scale"):
                state[key] = torch.ones_like(value)
        mine.load_state_dict(state)
    else:
        print("loading official mtp weights (dequantized) ...")
        state = load_official_mtp_state()
        missing, unexpected = mine.load_state_dict(state, strict=False)
        assert not missing, f"missing keys: {missing[:8]}"
        assert not unexpected, f"unexpected keys: {unexpected[:8]}"

    print("building reference blocks ...")
    ref = load_reference_module()
    blocks = build_reference_blocks(ref, cfg, n_target_layers, S + block + 8)
    for stage in range(cfg.num_hidden_layers):
        prefix = f"mtp.{stage}."
        sub = {
            k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)
        }
        missing, unexpected = blocks[stage].load_state_dict(sub, strict=False)
        missing = [m for m in missing if "kv_cache" not in m]
        assert not missing, f"stage {stage} missing: {missing[:8]}"
        assert not unexpected, f"stage {stage} unexpected: {unexpected[:8]}"
    blocks.eval()

    # ----- identical inputs -----
    dim = cfg.hidden_size
    features = (torch.randn(1, S, n_features * dim) * 2.0).to(torch.bfloat16)
    noise = (torch.randn(1, block, dim) * 0.5).to(torch.bfloat16)
    anchor = S  # the draft block starts right after the captured context

    # ----- SpecForge forward -----
    with torch.no_grad():
        mine_out = mine(
            noise_embedding=noise,
            target_hidden=features,
            anchor_positions=torch.tensor([[anchor]]),
            block_keep_mask=torch.ones(1, 1, dtype=torch.bool),
        )
        mine_prenorm = mine.pop_confidence_hidden()

    # ----- reference forward (prefill ring cache, then one draft block) -----
    with torch.no_grad():
        stage0 = blocks[0]
        main_x_full = stage0.main_norm(stage0.main_proj(features))  # [1, S, dim]
        hc = cfg.hc_mult
        x = noise.unsqueeze(2).repeat(1, 1, hc, 1)
        dummy_ids = torch.zeros(1, block, dtype=torch.long)
        # Prefill positions 0..S-2 into each stage's ring KV cache.
        for blk in blocks:
            blk.attn(x, 0, main_x_full[:, : S - 1])
        # Decode-style draft block at start_pos = S-1 (anchor = S).
        h = x
        for blk in blocks:
            h = blk(h, S - 1, dummy_ids, main_x_full[:, S - 1 : S])
        last = blocks[-1]
        ref_prenorm = last.hc_head(
            h, last.hc_head_fn, last.hc_head_scale, last.hc_head_base
        )
        ref_out = last.norm(ref_prenorm)

    # ----- compare -----
    def report(name, a, b):
        a = a.float().reshape(block, -1)
        b = b.float().reshape(block, -1)
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        rel = (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6)
        print(f"{name}: cos per position {[round(c, 6) for c in cos.tolist()]}")
        print(f"{name}: rel-err per position {[round(r, 6) for r in rel.tolist()]}")
        return cos.min().item(), rel.max().item()

    cos1, rel1 = report("prenorm hidden", mine_prenorm, ref_prenorm)
    cos2, rel2 = report("postnorm hidden", mine_out, ref_out)

    # The 256-expert top-6 MoE is discretely sensitive: a ~0.3% numeric
    # difference (bf16 reduction order) can flip one expert for one block
    # position and swing that position's hidden by ~10-20% while every other
    # position matches to <5%. Verified input-dependent (the outlier position
    # moves with the RNG seed) with the computation up to the routing decision
    # float-exact — so the gate tolerates one routing-flip outlier per run.
    def passes(out_mine, out_ref):
        a = out_mine.float().reshape(cfg.block_size, -1)
        b = out_ref.float().reshape(cfg.block_size, -1)
        rel = ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6)).tolist()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).tolist()
        outliers = sum(1 for r in rel if r >= 0.06)
        return (
            outliers <= 1
            and all(c > 0.97 for c in cos)
            and sorted(rel)[len(rel) // 2] < 0.05
        )

    ok = passes(mine_prenorm, ref_prenorm) and passes(mine_out, ref_out)
    print("PARITY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
