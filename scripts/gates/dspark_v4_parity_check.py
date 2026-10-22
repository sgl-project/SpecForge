#!/usr/bin/env python3
"""Gate 0: parity of ``DSparkV4DraftModel`` vs the official reference impl.

Loads the same weights into BOTH:
  1. SpecForge's training-side ``DSparkV4DraftModel`` (parallel-block forward),
  2. the reference ``inference/model.py`` ``DSparkBlock`` stack shipped with
     DeepSeek-V4-Flash-0731 (decode-style forward with a ring KV cache), with
     its tilelang kernels shimmed in torch,
then runs one draft block through each on identical inputs and compares, per
block position, the MoE routing of every stage and the pre-norm (confidence)
and post-norm hidden states.

Weights are the official dequantized ``mtp.*`` weights of ``--snapshot``, or a
random tiny model with ``--tiny``; the reference code is
``<snapshot>/inference``.

Two kinds of check:

- **Exact** (``--fp32`` with ``--tiny`` or ``--no-fake-quant``): fp32
  differences between the two implementations stay at rounding level, so
  every stage's attention and MoE output and the final hidden states must
  agree to rel-err < 1e-4 with no routing flip. ``--tiny --fp32`` keeps the
  FP8 activation fake quantization on and so also pins where it sits (both
  sides round identically at those sizes); with official weights it has to
  be off, because that step function turns fp32 rounding differences into
  whole FP8 steps.
- **Serving dtype** (everything else, by default bf16 with fake
  quantization): a top-k MoE is discretely sensitive, so a rounding
  difference can flip one expert for one token and that position then
  legitimately differs. Routing is compared first: at least
  ``--min-routing-agreement`` of all (stage, position) routing decisions must
  agree (a broken router agrees almost nowhere; DeepSeek-V4's selection
  biases make close top-k calls, so bf16 flips a few, mostly in the last
  stage), and positions whose routing agrees in every stage must match
  (cos > 0.99, rel-err < 10%).

Run on CPU; no GPU or sglang required:
    python3 scripts/gates/dspark_v4_parity_check.py \
        [--fp32 --tiny | --fp32 --no-fake-quant] [--snapshot <dir>]
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

#: Exact checks differ by floating-point rounding only.
EXACT_TOLERANCE = 1e-4
#: Serving-dtype checks: bf16 noise through three stages and the FP8 KV fake
#: quantization.
SERVING_MIN_COS, SERVING_MAX_REL = 0.99, 0.10

DRAFT_CONFIG = "configs/deepseek-v4-flash-dspark-official.json"
NOISE_TOKEN_ID = 128799

# ---------------------------------------------------------------------------
# torch shims for the reference tilelang kernels
# ---------------------------------------------------------------------------


def _shim_act_quant(x, block_size=128, scale_fmt=None, scale_dtype=None, inplace=False):
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
    gathered = kv[torch.arange(b).view(b, 1, 1), idx.clamp(min=0)]  # [b,s,topk,d]
    scores = torch.einsum("bshd,bskd->bhsk", q.float(), gathered.float())
    scores = scores * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(1), torch.finfo(torch.float32).min)
    sink = attn_sink.float().view(1, h, 1, 1).expand(b, -1, s, 1)
    probs = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
    out = torch.einsum("bhsk,bskd->bshd", probs, gathered.float())
    return out.to(q.dtype)


def _shim_hc_split_sinkhorn(
    mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6
):
    from specforge.modeling.draft.dspark_v4 import _hc_split_sinkhorn

    return _hc_split_sinkhorn(
        mixes.float(), hc_scale, hc_base, hc_mult, sinkhorn_iters, eps
    )


def _unused(name):
    def fail(*args, **kwargs):
        raise NotImplementedError(f"{name} is not needed for DSpark parity")

    return fail


def _no_act_quant(x, *args, **kwargs):
    return x


def load_reference_module(inference_dir, fake_quant=True):
    kernel = types.ModuleType("kernel")
    kernel.act_quant = _shim_act_quant if fake_quant else _no_act_quant
    kernel.fp4_act_quant = _unused("fp4_act_quant")
    kernel.fp8_gemm = _unused("fp8_gemm")
    kernel.fp4_gemm = _unused("fp4_gemm")
    kernel.sparse_attn = _shim_sparse_attn
    kernel.hc_split_sinkhorn = _shim_hc_split_sinkhorn
    sys.modules["kernel"] = kernel

    path = os.path.join(inference_dir, "model.py")
    spec = importlib.util.spec_from_file_location("dspark_reference_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dspark_reference_model"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------


def build_config(tiny: bool):
    from specforge.modeling.auto import AutoDraftModelConfig

    cfg = AutoDraftModelConfig.from_file(os.path.join(REPO_ROOT, DRAFT_CONFIG))
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
        cfg.dflash_config = {**cfg.dflash_config, "markov_rank": 16}
    return cfg


def _reference_args(ref, cfg, n_layers, max_seq_len):
    return ref.ModelArgs(
        max_batch_size=1,
        max_seq_len=max_seq_len,
        dtype="bf16",
        expert_dtype=None,
        vocab_size=cfg.vocab_size,
        dim=cfg.hidden_size,
        moe_inter_dim=cfg.moe_intermediate_size,
        n_layers=n_layers,
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
        dspark_noise_token_id=NOISE_TOKEN_ID,
        dspark_target_layer_ids=tuple(cfg.dflash_config["target_layer_ids"]),
        dspark_markov_rank=cfg.dflash_config["markov_rank"],
        scale_fmt="ue8m0",
        scale_dtype="fp32",
        n_hash_layers=0,
    )


def build_reference_blocks(ref, cfg, n_layers, max_seq_len, dtype):
    args = _reference_args(ref, cfg, n_layers, max_seq_len)
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
    if dtype == torch.float32:
        blocks.float()
    return blocks


def _official_names(state, mine):
    """Module-tree state of ``mine`` -> the official checkpoint naming."""
    from specforge.modeling.draft.moe import to_checkpoint_state_dict

    last = f"mtp.{mine.config.num_hidden_layers - 1}."
    out = {}
    for key, value in to_checkpoint_state_dict(state, mine).items():
        if key.startswith(("markov_head.", "confidence_head.")):
            key = last + key
        out[key] = value
    return out


def _record_routing(gate, sink, reference):
    def hook(module, args, output):
        indices = output[1] if reference else output.indices
        sink.append(indices.detach().reshape(-1, indices.shape[-1]))

    return gate.register_forward_hook(hook)


def _record_output(module, sink, name, decode_only=False):
    """Record ``module``'s output per block position (the reference attention
    also runs at prefill, start_pos == 0, which is skipped)."""

    def hook(module, args, output):
        if decode_only and args[1] == 0:
            return
        sink.setdefault(name, output.detach().float().reshape(-1, output.shape[-1]))

    return module.register_forward_hook(hook)


def _rel_err(a, b):
    return ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-12)).max().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=None, help="official checkpoint dir")
    parser.add_argument(
        "--reference-dir", default=None, help="default: <snapshot>/inference"
    )
    parser.add_argument("--context-len", type=int, default=200)
    parser.add_argument(
        "--tiny", action="store_true", help="random tiny config instead of official"
    )
    parser.add_argument(
        "--fp32", action="store_true", help="fp32 structure check (default: bf16)"
    )
    parser.add_argument(
        "--no-fake-quant",
        action="store_true",
        help="turn the FP8 activation fake quantization off on both sides",
    )
    parser.add_argument("--min-routing-agreement", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    snapshot = args.snapshot or os.environ.get(
        "DSPARK_OFFICIAL_SNAPSHOT", DEFAULT_SNAPSHOT
    )
    reference_dir = args.reference_dir or os.path.join(snapshot, "inference")
    dtype = torch.float32 if args.fp32 else torch.bfloat16

    cfg = build_config(args.tiny)
    block = cfg.block_size
    n_features = len(cfg.dflash_config["target_layer_ids"])
    S = args.context_len

    from specforge.modeling.auto import AutoDraftModel
    from specforge.modeling.draft.moe import from_checkpoint_state_dict

    print(f"building SpecForge model ({dtype}) ...")
    mine = AutoDraftModel.from_config(cfg, torch_dtype=dtype).eval()
    # The reference computes the weightless q rms-scale in the activation
    # dtype; sglang serving (the real target) uses fp32. Use the reference
    # math here so the gate isolates structural bugs.
    for stage in mine.mtp:
        stage.attn.qscale_ref_compat = True
    if args.no_fake_quant:
        mine.kv_fake_quant = False

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
        official = _official_names(state, mine)
    else:
        print(f"loading official mtp weights from {snapshot} (dequantized) ...")
        official = load_official_mtp_state(snapshot)
        result = mine.load_state_dict(
            from_checkpoint_state_dict(official, mine), strict=False
        )
        assert not result.missing_keys, f"missing keys: {result.missing_keys[:8]}"
        assert not result.unexpected_keys, f"unexpected: {result.unexpected_keys[:8]}"

    print(f"building reference blocks from {reference_dir} ...")
    ref = load_reference_module(reference_dir, fake_quant=not args.no_fake_quant)
    blocks = build_reference_blocks(
        ref, cfg, cfg.num_target_layers, S + block + 8, dtype
    )
    for stage in range(cfg.num_hidden_layers):
        prefix = f"mtp.{stage}."
        sub = {k[len(prefix) :]: v for k, v in official.items() if k.startswith(prefix)}
        missing, unexpected = blocks[stage].load_state_dict(sub, strict=False)
        missing = [m for m in missing if "kv_cache" not in m]
        assert not missing, f"stage {stage} missing: {missing[:8]}"
        assert not unexpected, f"stage {stage} unexpected: {unexpected[:8]}"
    blocks.eval()
    del official

    mine_routes, ref_routes = [], []
    mine_parts, ref_parts = {}, {}
    for i, (stage, blk) in enumerate(zip(mine.mtp, blocks)):
        _record_routing(stage.ffn.gate, mine_routes, reference=False)
        _record_routing(blk.ffn.gate, ref_routes, reference=True)
        _record_output(stage.attn, mine_parts, f"stage {i} attention")
        _record_output(blk.attn, ref_parts, f"stage {i} attention", decode_only=True)
        _record_output(stage.ffn, mine_parts, f"stage {i} moe")
        _record_output(blk.ffn, ref_parts, f"stage {i} moe")

    # ----- identical inputs -----
    dim = cfg.hidden_size
    features = (torch.randn(1, S, n_features * dim) * 2.0).to(dtype)
    noise = (torch.randn(1, block, dim) * 0.5).to(dtype)
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
        x = noise.unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        # Prefill positions 0..S-2 into each stage's ring KV cache.
        for blk in blocks:
            blk.attn(x, 0, main_x_full[:, : S - 1])
        # Decode-style draft block at start_pos = S-1 (anchor = S).
        dummy_ids = torch.zeros(1, block, dtype=torch.long)
        h = x
        for blk in blocks:
            h = blk(h, S - 1, dummy_ids, main_x_full[:, S - 1 : S])
        last = blocks[-1]
        ref_prenorm = last.hc_head(
            h, last.hc_head_fn, last.hc_head_scale, last.hc_head_base
        )
        ref_out = last.norm(ref_prenorm)

    # ----- compare -----
    agree = [True] * block
    flips = 0
    for stage, (a, b) in enumerate(zip(mine_routes, ref_routes)):
        for p in range(block):
            if set(a[p].tolist()) != set(b[p].tolist()):
                agree[p] = False
                flips += 1
                print(
                    f"stage {stage} position {p}: routing flip "
                    f"{sorted(a[p].tolist())} vs reference {sorted(b[p].tolist())}"
                )
    decisions = len(mine_routes) * block
    agreement = 1.0 - flips / decisions
    print(
        f"routing agrees on {decisions - flips}/{decisions} (stage, position) "
        f"decisions; {sum(agree)}/{block} positions agree in every stage"
    )
    # Sublayer outputs, before the residual streams dilute a difference.
    part_errors = {
        name: _rel_err(mine_parts[name], ref_parts[name]) for name in ref_parts
    }
    for name, err in part_errors.items():
        print(f"{name}: max rel-err {err:.2e}")

    def per_position(name, a, b):
        a = a.float().reshape(block, -1)
        b = b.float().reshape(block, -1)
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).tolist()
        rel = ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6)).tolist()
        print(f"{name}: cos per position {[round(c, 6) for c in cos]}")
        print(f"{name}: rel-err per position {[f'{r:.2e}' for r in rel]}")
        return cos, rel

    results = [
        per_position("prenorm hidden", mine_prenorm, ref_prenorm),
        per_position("postnorm hidden", mine_out, ref_out),
    ]
    exact = args.fp32 and (args.tiny or args.no_fake_quant)
    if exact:
        ok = (
            flips == 0
            and all(r < EXACT_TOLERANCE for _, rel in results for r in rel)
            and all(err < EXACT_TOLERANCE for err in part_errors.values())
        )
    else:
        ok = agreement >= args.min_routing_agreement and all(
            c > SERVING_MIN_COS and r < SERVING_MAX_REL
            for cos, rel in results
            for c, r, same in zip(cos, rel, agree)
            if same
        )
    print(
        f"PARITY ({'exact' if exact else 'serving dtype'}):", "PASS" if ok else "FAIL"
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
