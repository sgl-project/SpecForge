#!/usr/bin/env python3
"""Build a warm-start checkpoint for DSparkV4DraftModel.

Two modes:
  --from-target (default): initialize the three mtp stages from the TARGET
    model's own layers 40-42 (dequantized to bf16), the last stage's final
    norm / hc_head from the target's top-level norm / hc_head, an identity
    main_proj on the last captured feature, and fresh markov/confidence heads.
    This is the "train from scratch, but target-initialized" warm start.
  --from-official: load the official DSpark drafter's own mtp.* weights
    (dequantized). Used only by validation gates - NOT for training runs.

Output: an HF-style draft dir (config.json + model.safetensors) consumable via
``model.draft_checkpoint_path``.

Example:
    python3 scripts/init_dspark_v4_from_target.py \
        --output-dir outputs/dspark-v4-official-init
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from dspark_v4_official_weights import (  # noqa: E402
    dequant_fp4_packed,
    dequant_fp8_block,
    iter_official_tensors,
    load_official_mtp_state,
)

TARGET_SNAPSHOT = (
    "/cluster-storage/models/models--deepseek-ai--DeepSeek-V4-Flash-0731/"
    "snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062"
)
DRAFT_CONFIG = os.path.join(REPO_ROOT, "configs/deepseek-v4-flash-dspark-official.json")

# Per-stage tensors copied verbatim from target layer (40 + stage).
_STAGE_SUFFIXES = (
    "attn.attn_sink",
    "attn.q_norm.weight",
    "attn.kv_norm.weight",
    "attn.wq_a.weight",
    "attn.wq_b.weight",
    "attn.wkv.weight",
    "attn.wo_a.weight",
    "attn.wo_b.weight",
    "attn_norm.weight",
    "ffn_norm.weight",
    "ffn.gate.weight",
    "ffn.gate.bias",
    "hc_attn_fn",
    "hc_attn_base",
    "hc_attn_scale",
    "hc_ffn_fn",
    "hc_ffn_base",
    "hc_ffn_scale",
)


def _dequant(raw: dict, name: str) -> torch.Tensor:
    tensor = raw[name]
    scale_name = name[: -len(".weight")] + ".scale" if name.endswith(".weight") else None
    if tensor.dtype == torch.float8_e4m3fn:
        return dequant_fp8_block(tensor, raw[scale_name])
    if tensor.dtype == torch.int8:
        return dequant_fp4_packed(tensor, raw[scale_name])
    return tensor


def _load_target_tensors(snapshot: str, prefixes: tuple[str, ...]) -> dict:
    raw = {}
    for name, tensor in iter_official_tensors(snapshot, prefix=""):
        if name.startswith(prefixes):
            raw[name] = tensor
    return raw


def build_target_init_state(
    snapshot: str,
    layer_ids: list[int],
    n_experts: int,
    hidden_size: int,
    markov_rank: int,
    vocab_size: int,
    main_proj_init: str,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    n_stages = len(layer_ids)
    prefixes = tuple(f"layers.{lid}." for lid in layer_ids) + (
        "norm.weight",
        "hc_head_fn",
        "hc_head_base",
        "hc_head_scale",
    )
    print(f"reading target tensors for layers {layer_ids} ...")
    raw = _load_target_tensors(snapshot, prefixes)

    state = {}
    for stage, lid in enumerate(layer_ids):
        src = f"layers.{lid}."
        dst = f"mtp.{stage}."
        for suffix in _STAGE_SUFFIXES:
            state[dst + suffix] = _dequant(raw, src + suffix)
        for expert in range(n_experts):
            for w in ("w1", "w2", "w3"):
                suffix = f"ffn.experts.{expert}.{w}.weight"
                state[dst + suffix] = _dequant(raw, src + suffix)
        for w in ("w1", "w2", "w3"):
            suffix = f"ffn.shared_experts.{w}.weight"
            state[dst + suffix] = _dequant(raw, src + suffix)
        print(f"  stage {stage} <- target layer {lid}")

    last = f"mtp.{n_stages - 1}."
    state[last + "norm.weight"] = raw["norm.weight"]
    state[last + "hc_head_fn"] = raw["hc_head_fn"]
    state[last + "hc_head_base"] = raw["hc_head_base"]
    state[last + "hc_head_scale"] = raw["hc_head_scale"]
    del last  # heads below use the model's root (tree) naming

    # main_proj: identity on the last captured feature (or an average).
    proj = torch.zeros(hidden_size, hidden_size * n_stages)
    eye = torch.eye(hidden_size)
    if main_proj_init == "average":
        for i in range(n_stages):
            proj[:, i * hidden_size : (i + 1) * hidden_size] = eye / n_stages
    else:
        proj[:, (n_stages - 1) * hidden_size :] = eye
    state["mtp.0.main_proj.weight"] = proj.to(torch.bfloat16)
    state["mtp.0.main_norm.weight"] = torch.ones(hidden_size, dtype=torch.bfloat16)

    # Fresh heads: LoRA-style (random w1, zero w2/confidence).
    state["markov_head.markov_w1.weight"] = (
        torch.randn(vocab_size, markov_rank) * 0.02
    ).to(torch.bfloat16)
    state["markov_head.markov_w2.weight"] = torch.zeros(
        vocab_size, markov_rank, dtype=torch.bfloat16
    )
    state["confidence_head.proj.weight"] = torch.zeros(
        1, hidden_size + markov_rank, dtype=torch.bfloat16
    )
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-snapshot", default=TARGET_SNAPSHOT)
    parser.add_argument("--draft-config", default=DRAFT_CONFIG)
    parser.add_argument(
        "--from-official",
        action="store_true",
        help="use the official drafter weights (validation gates only)",
    )
    parser.add_argument(
        "--main-proj-init", choices=("identity-last", "average"),
        default="identity-last",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.draft_config) as stream:
        cfg = json.load(stream)
    layer_ids = list(cfg["dflash_config"]["target_layer_ids"])

    if args.from_official:
        print("loading official drafter weights ...")
        state = load_official_mtp_state()
        # The model owns the heads at its root (tree naming); official
        # checkpoints keep them under the last stage.
        last = int(cfg["num_hidden_layers"]) - 1
        for head in ("markov_head.", "confidence_head."):
            prefix = f"mtp.{last}.{head}"
            for key in [k for k in state if k.startswith(prefix)]:
                state[key[len(f"mtp.{last}."):]] = state.pop(key)
    else:
        state = build_target_init_state(
            snapshot=args.target_snapshot,
            layer_ids=layer_ids,
            n_experts=int(cfg["n_routed_experts"]),
            hidden_size=int(cfg["hidden_size"]),
            markov_rank=int(cfg["dflash_config"]["markov_rank"]),
            vocab_size=int(cfg["vocab_size"]),
            main_proj_init=args.main_proj_init,
            seed=args.seed,
        )

    # Validate against the actual model skeleton before writing anything.
    print("validating against DSparkV4DraftModel ...")
    from specforge.modeling.auto import AutoDraftModel
    from specforge.training.model_loading import load_draft_config_source

    draft_config = load_draft_config_source(args.draft_config)
    with torch.device("meta"):
        model = AutoDraftModel.from_config(draft_config)
    expected = model.state_dict()  # official naming (state-dict hooks)
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    assert not missing, f"missing: {missing[:8]}"
    assert not unexpected, f"unexpected: {unexpected[:8]}"
    for key, value in state.items():
        assert tuple(value.shape) == tuple(expected[key].shape), (
            key, tuple(value.shape), tuple(expected[key].shape),
        )

    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copyfile(
        args.draft_config, os.path.join(args.output_dir, "config.json")
    )
    from safetensors.torch import save_file

    out = os.path.join(args.output_dir, "model.safetensors")
    print(f"writing {out} ...")
    save_file({k: v.contiguous() for k, v in state.items()}, out)
    total = sum(v.numel() for v in state.values())
    print(f"done: {len(state)} tensors, {total/1e9:.2f}B params")


if __name__ == "__main__":
    main()
