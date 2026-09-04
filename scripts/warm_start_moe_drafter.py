#!/usr/bin/env python3
"""Build a warm-start source for an MoE DFlash-family drafter from a DeepSeek-V4 target.

Constructs the draft model from its config (random init for attention, heads,
norms, projections), seeds every MoE layer's routed experts, gate weight,
noaux bias and shared expert from one target layer (dequantized fp4/fp8 ->
bf16), and writes an HF-format directory usable as
``model.draft_checkpoint_path``. Requires the draft's expert shape to match the
target's (256 x 2048 for DeepSeek-V4-Flash); a smaller draft takes a strided
subset of experts.

Example:
  python scripts/warm_start_moe_drafter.py \
      --draft-config examples/configs/kan-ablations/deepseek-v4-flash-dspark-moe256-auxbal.json \
      --target-snapshot /cluster-storage/models/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/<sha> \
      --target-layers 3,11,21,31,41 --output-dir warm-starts/moe256-from-0731
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--draft-config", required=True)
    ap.add_argument(
        "--target-snapshot", required=True, help="local HF snapshot dir of the target"
    )
    ap.add_argument(
        "--target-layers",
        required=True,
        help="comma list, one target layer per draft layer",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--select",
        default="strided",
        help="expert selection when draft has fewer experts",
    )
    args = ap.parse_args()

    from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
    from specforge.modeling.draft.moe import (
        apply_warm_start,
        iter_moe_layers,
        plan_warm_start,
        resolve_moe_config,
        to_checkpoint_state_dict,
    )
    from specforge.modeling.draft.moe.deepseek_v4_target import load_target_moe_layer

    target_layers = [int(x) for x in args.target_layers.split(",")]
    config = AutoDraftModelConfig.from_file(args.draft_config)
    moe_cfg = resolve_moe_config(config)
    if moe_cfg is None:
        raise SystemExit("draft config is dense (n_routed_experts == 0)")
    torch.manual_seed(args.seed)
    t0 = time.time()
    model = AutoDraftModel.from_config(config, torch_dtype=torch.bfloat16)
    layers = list(iter_moe_layers(model))
    if len(layers) != len(target_layers):
        raise SystemExit(
            f"{len(layers)} MoE layers but {len(target_layers)} target layers given"
        )
    print(
        f"built draft ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params) in {time.time()-t0:.0f}s"
    )

    target_cfg = json.load(open(os.path.join(args.target_snapshot, "config.json")))
    n_target = int(target_cfg["n_routed_experts"])
    for i, (layer, tl) in enumerate(zip(layers, target_layers)):
        t1 = time.time()
        source = load_target_moe_layer(args.target_snapshot, tl)
        plan = plan_warm_start(
            layer.cfg, n_target_experts=n_target, strategy=args.select
        )
        loaded = apply_warm_start(layer, plan, source)
        print(
            f"draft layer {i} <- target layer {tl}: {len(loaded)} tensors, "
            f"experts {plan.target_expert_ids[:3]}...{plan.target_expert_ids[-1]} ({time.time()-t1:.0f}s)"
        )

    for key, value in moe_cfg.serving_fields().items():
        setattr(model.config, key, value)
    model.config.moe_warm_start = {
        "target": os.path.basename(
            os.path.dirname(os.path.dirname(args.target_snapshot.rstrip("/")))
        ),
        "snapshot": os.path.basename(args.target_snapshot.rstrip("/")),
        "target_layers": target_layers,
        "select": args.select,
        "seed": args.seed,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(
        args.output_dir, state_dict=to_checkpoint_state_dict(model.state_dict())
    )
    print(f"wrote {args.output_dir} in {time.time()-t0:.0f}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
