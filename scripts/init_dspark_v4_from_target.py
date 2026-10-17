#!/usr/bin/env python3
"""Build a warm-start checkpoint for DSparkV4DraftModel.

Two modes:
  --from-target (default): initialize the mtp stages from the TARGET model's
    own layers (the draft config's target_layer_ids unless --stage-layers is
    given), dequantized to bf16: attention, norms, hyper-connections and the
    MoE (a draft with fewer routed experts than the target takes an evenly
    strided subset, with the matching router rows). The last stage's final
    norm and hc_head come from the target's top-level ones;
    main_proj starts as the identity on the last captured feature, and the
    Markov/confidence heads start fresh. This is the "train from scratch, but
    target-initialized" warm start.
  --from-official: load the official DSpark drafter's own mtp.* weights
    (dequantized). Used by validation gates and by fine-tuning runs that start
    from the released drafter.

Output: an HF-style draft dir (config.json + model.safetensors) in the
official tensor naming, consumable via ``model.draft_checkpoint_path``.

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
    DEFAULT_SNAPSHOT,
    load_official_mtp_state,
)

from specforge.modeling.draft.moe.checkpoint import select_target_experts  # noqa: E402
from specforge.modeling.draft.moe.families.deepseek import (  # noqa: E402
    dequantize_tensors,
    iter_checkpoint_tensors,
    read_config,
    select_deepseek_experts,
)

DRAFT_CONFIG = os.path.join(REPO_ROOT, "configs/deepseek-v4-flash-dspark-official.json")

# Per-stage tensors copied verbatim from the stage's target layer.
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
    "hc_attn_fn",
    "hc_attn_base",
    "hc_attn_scale",
    "hc_ffn_fn",
    "hc_ffn_base",
    "hc_ffn_scale",
)
_HC_HEAD = ("hc_head_fn", "hc_head_base", "hc_head_scale")


def _read(snapshot: str, prefixes) -> dict:
    """Dequantized target tensors under ``prefixes`` (only their shards are read)."""
    raw = {}
    for prefix in prefixes:
        raw.update(iter_checkpoint_tensors(snapshot, prefix))
    return dequantize_tensors(raw)


def build_target_init_state(
    snapshot: str,
    layer_ids: list[int],
    n_experts: int,
    *,
    n_features: int | None = None,
    hidden_size: int,
    markov_rank: int,
    vocab_size: int,
    main_proj_init: str,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    n_stages = len(layer_ids)
    # Number of captured target features feeding main_proj; defaults to one
    # per stage (the official last-3 layout). A spread capture (e.g. 5 layers
    # [1, 11, 21, 31, 41]) still initializes the 3 stages from layers 40-42.
    n_features = n_stages if n_features is None else int(n_features)
    n_target = int(read_config(snapshot)["n_routed_experts"])
    expert_ids = select_target_experts(n_target, n_experts)
    top = ("norm.weight",) + _HC_HEAD
    print(f"reading target tensors for layers {layer_ids} ...")
    raw = _read(snapshot, tuple(f"layers.{lid}." for lid in layer_ids) + top)

    state = {}
    for stage, lid in enumerate(layer_ids):
        src = f"layers.{lid}."
        dst = f"mtp.{stage}."
        for suffix in _STAGE_SUFFIXES:
            state[dst + suffix] = raw[src + suffix]
        moe = {
            key[len(src + "ffn.") :]: value
            for key, value in raw.items()
            if key.startswith(src + "ffn.")
        }
        if "gate.tid2eid" in moe:
            raise SystemExit(
                f"target layer {lid} routes by token hash and has no learned "
                "router; pick layers at or above num_hash_layers"
            )
        for key, value in select_deepseek_experts(moe, expert_ids).items():
            state[dst + "ffn." + key] = value
        print(f"  stage {stage} <- target layer {lid}")

    last = f"mtp.{n_stages - 1}."
    for name in top:
        state[last + name] = raw[name]
    del last  # heads below use the model's root (tree) naming

    # main_proj: identity on the last captured feature (or an average).
    proj = torch.zeros(hidden_size, n_features * hidden_size)
    eye = torch.eye(hidden_size)
    if main_proj_init == "average":
        for i in range(n_features):
            proj[:, i * hidden_size : (i + 1) * hidden_size] = eye / n_features
    else:
        proj[:, (n_features - 1) * hidden_size :] = eye
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
    parser.add_argument("--target-snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--draft-config", default=DRAFT_CONFIG)
    parser.add_argument(
        "--from-official",
        action="store_true",
        help="use the official drafter weights of --target-snapshot",
    )
    parser.add_argument(
        "--main-proj-init",
        choices=("identity-last", "average"),
        default="identity-last",
    )
    parser.add_argument(
        "--stage-layers",
        default=None,
        help="comma-separated target layers to copy into the mtp stages "
        "(default: the draft config's target_layer_ids; use 40,41,42 for a "
        "spread-capture draft whose captured layers differ from its stages)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
    from specforge.modeling.draft.dspark_v4 import rename_official_heads
    from specforge.modeling.draft.moe import to_checkpoint_state_dict

    with open(args.draft_config) as stream:
        cfg = json.load(stream)
    draft_config = AutoDraftModelConfig.from_file(args.draft_config)
    capture_ids = list(cfg["dflash_config"]["target_layer_ids"])
    layer_ids = (
        [int(x) for x in args.stage_layers.split(",")]
        if args.stage_layers
        else capture_ids
    )
    if len(layer_ids) != int(cfg["num_hidden_layers"]):
        raise SystemExit(
            f"{len(layer_ids)} stage source layers for a "
            f"{cfg['num_hidden_layers']}-stage draft; pass --stage-layers"
        )

    if args.from_official:
        print("loading official drafter weights ...")
        state = load_official_mtp_state(args.target_snapshot)
        # The model owns the heads at its root (tree naming); official
        # checkpoints keep them under the last stage.
        rename_official_heads(state, int(cfg["num_hidden_layers"]) - 1)
    else:
        state = build_target_init_state(
            snapshot=args.target_snapshot,
            layer_ids=layer_ids,
            n_experts=int(cfg["n_routed_experts"]),
            n_features=len(capture_ids),
            hidden_size=int(cfg["hidden_size"]),
            markov_rank=int(cfg["dflash_config"]["markov_rank"]),
            vocab_size=int(cfg["vocab_size"]),
            main_proj_init=args.main_proj_init,
            seed=args.seed,
        )

    # Validate against the actual model skeleton before writing anything:
    # checkpoint files name the MoE tensors through the deepseek layout.
    print("validating against DSparkV4DraftModel ...")
    with torch.device("meta"):
        model = AutoDraftModel.from_config(draft_config)
    expected = to_checkpoint_state_dict(model.state_dict(), model)
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    assert not missing, f"missing: {missing[:8]}"
    assert not unexpected, f"unexpected: {unexpected[:8]}"
    for key, value in state.items():
        assert tuple(value.shape) == tuple(expected[key].shape), (
            key,
            tuple(value.shape),
            tuple(expected[key].shape),
        )

    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copyfile(args.draft_config, os.path.join(args.output_dir, "config.json"))
    from safetensors.torch import save_file

    out = os.path.join(args.output_dir, "model.safetensors")
    print(f"writing {out} ...")
    save_file({k: v.contiguous() for k, v in state.items()}, out)
    total = sum(v.numel() for v in state.values())
    print(f"done: {len(state)} tensors, {total/1e9:.2f}B params")


if __name__ == "__main__":
    main()
