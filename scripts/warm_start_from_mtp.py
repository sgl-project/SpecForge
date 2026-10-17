"""Build a Qwen3.8-Flash-Next DSpark warm start from the target's MTP layer.

Writes a pretrained draft directory (config.json + safetensors) whose attention
and MoE come from ``mtp.layers.0`` and whose remaining tensors -- input
projection, layer norms, Markov and confidence heads -- are freshly
initialized. Train from it with ``model.draft_checkpoint_path=<output-dir>``.

  python scripts/warm_start_from_mtp.py \\
      --target-model-path /path/to/Qwen3.8-Flash-Next \\
      --draft-model-config configs/qwen3.8-flash-next-dspark-moe64.json \\
      --expert-selection stride \\
      --output-dir outputs/flash-next-moe64-mtp-init

A drafter with fewer experts than the MTP layer (the 64-expert arm) must say
which ones it starts from: ``--expert-ids`` (a comma list or a JSON file of
ids), or ``--expert-selection stride`` for evenly spaced ids.

The mapping is in ``specforge/modeling/draft/mtp_warm_start.py``. It needs no
accelerator, only host memory: the drafter itself, built directly in the
requested dtype (24.3 GiB in BF16 for the 512-expert arm, 3.7 GiB for 64), plus
the MTP layer read from the target (about 4.8 GiB). Plan for about 32 GiB at
peak for the 512-expert arm.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

import torch

from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
from specforge.modeling.draft.mtp_warm_start import (
    apply_mtp_warm_start,
    read_mtp_tensors,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--draft-model-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--layers",
        default="all",
        help="drafter layers to seed: 'all' or a comma list such as '0'",
    )
    parser.add_argument(
        "--expert-ids",
        default=None,
        help="MTP expert ids the drafter's experts start from, in order: a "
        "comma list or a JSON file holding a list",
    )
    parser.add_argument(
        "--expert-selection",
        choices=("all", "stride"),
        default="all",
        help="without --expert-ids: 'all' needs as many drafter experts as the "
        "MTP layer has; 'stride' takes evenly spaced ids",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    return parser.parse_args(argv)


def resolve_expert_ids(
    args, *, draft_experts: int, mtp_experts: int
) -> Optional[List[int]]:
    if args.expert_ids:
        if os.path.isfile(args.expert_ids):
            with open(args.expert_ids, encoding="utf-8") as stream:
                return [int(i) for i in json.load(stream)]
        return [int(i) for i in args.expert_ids.split(",") if i.strip()]
    if args.expert_selection == "stride":
        step = mtp_experts / draft_experts
        return [int(i * step) for i in range(draft_experts)]
    if draft_experts != mtp_experts:
        raise SystemExit(
            f"the drafter has {draft_experts} experts and the MTP layer "
            f"{mtp_experts}; pass --expert-ids or --expert-selection stride"
        )
    return None


def main(argv=None) -> None:
    args = parse_args(argv)
    dtype = getattr(torch, args.dtype)
    config = AutoDraftModelConfig.from_file(args.draft_model_config)
    tensors = read_mtp_tensors(args.target_model_path)
    expert_ids = resolve_expert_ids(
        args,
        draft_experts=int(getattr(config, "num_experts", 0) or 0),
        mtp_experts=int(tensors["mlp.gate.weight"].shape[0]),
    )
    layers = None if args.layers == "all" else [int(i) for i in args.layers.split(",")]

    torch.manual_seed(args.seed)
    model = AutoDraftModel.from_config(config, torch_dtype=dtype)
    written = apply_mtp_warm_start(model, tensors, layers=layers, expert_ids=expert_ids)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    manifest = {
        "source": os.path.abspath(args.target_model_path),
        "draft_model_config": os.path.abspath(args.draft_model_config),
        "layers": layers if layers is not None else list(range(len(model.layers))),
        "expert_ids": expert_ids,
        "seed": args.seed,
        "dtype": args.dtype,
        "tensors_from_mtp": len(written),
        "tensors_total": len(model.state_dict()),
    }
    with open(
        os.path.join(args.output_dir, "mtp_warm_start.json"), "w", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
