"""Capture text-only Qwen3.5 attention caches for offline DSpark-KV training.

Input JSONL rows contain already tokenized ``input_ids`` and ``loss_mask``.
This is teacher-forced capture, not response regeneration or speculative serving.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoConfig, Qwen3_5ForConditionalGeneration, Qwen3Config

from specforge.algorithms.dspark_kv.data import normalize_sample
from specforge.modeling.draft.target_kv import capture_geometry


@torch.inference_mode()
def capture_sample(text_model, input_ids, loss_mask, layer_ids):
    """Capture complete sequences; selected cache layers contain post-RoPE keys."""
    if input_ids.ndim != 1 or input_ids.dtype != torch.long or input_ids.numel() < 2:
        raise ValueError("input_ids must be a nonempty int64 sequence with >= 2 tokens")
    if loss_mask.shape != input_ids.shape:
        raise ValueError("loss_mask must match input_ids")
    device = next(text_model.parameters()).device
    ids = input_ids.to(device).unsqueeze(0)
    positions = torch.arange(ids.shape[1], device=device).unsqueeze(0)
    # A fresh cache per sample excludes previous samples and padding. For text,
    # Qwen3.5's three rotary axes share these same absolute positions.
    output = text_model(
        input_ids=ids, position_ids=positions, use_cache=True, return_dict=True
    )
    kv = []
    for layer_id in layer_ids:
        layer = output.past_key_values.layers[layer_id]
        keys, values = layer.keys, layer.values
        if (
            keys is None
            or values is None
            or keys.shape != values.shape
            or keys.shape[2] != ids.shape[1]
        ):
            raise ValueError(
                "capture requires full-attention K/V for the complete sequence"
            )
        kv.append(
            torch.stack((keys[0].transpose(0, 1), values[0].transpose(0, 1)), dim=1)
        )
    result = {
        "input_ids": input_ids.detach().cpu(),
        "loss_mask": loss_mask.detach().cpu(),
        "target_kv": torch.stack(kv, dim=1).cpu(),
        "target_last_hidden_states": output.last_hidden_state[0].cpu(),
    }
    normalize_sample(result, input_ids.numel())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--revision", help="Pin a public model revision for reproducibility"
    )
    parser.add_argument("--draft-config", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    args = parser.parse_args()
    if args.max_length < 2:
        parser.error("max-length must be >= 2")
    destination = Path(args.output_dir)
    if destination.exists() and any(destination.iterdir()):
        parser.error("output-dir must be empty to avoid mixing captures")
    draft = Qwen3Config.from_pretrained(args.draft_config)
    target_config = AutoConfig.from_pretrained(
        args.target_model, revision=args.revision
    )
    geometry = capture_geometry(draft, target_config)
    model = (
        Qwen3_5ForConditionalGeneration.from_pretrained(
            args.target_model,
            revision=args.revision,
            dtype=getattr(torch, args.dtype),
            attn_implementation="sdpa",
        )
        .to(args.device)
        .eval()
        .requires_grad_(False)
    )
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    input_hash = hashlib.sha256()
    with open(args.input_jsonl, "rb") as source:
        for line in source:
            input_hash.update(line)
            row = json.loads(line)
            if len(row["input_ids"]) != len(row["loss_mask"]):
                raise ValueError(
                    "input_ids and loss_mask must have equal length before truncation"
                )
            ids = row["input_ids"][: args.max_length]
            mask = row["loss_mask"][: args.max_length]
            if not isinstance(ids, list) or any(
                type(token) is not int for token in ids
            ):
                raise ValueError("input_ids must be a JSON integer array")
            if any(
                token < 0 or token >= target_config.text_config.vocab_size
                for token in ids
            ):
                raise ValueError("input token outside target vocabulary")
            features = capture_sample(
                model.model.language_model,
                torch.tensor(ids, dtype=torch.long),
                torch.tensor(mask, dtype=torch.float32),
                geometry["layer_ids"],
            )
            temporary = destination / f"sample-{count:08d}.tmp"
            torch.save(features, temporary)
            temporary.replace(destination / f"sample-{count:08d}.ckpt")
            count += 1
    if count == 0:
        raise ValueError("input-jsonl contains no samples")
    manifest = {
        "format": "dspark_target_kv_v1",
        "samples": count,
        "target_model": args.target_model,
        "target_revision": getattr(target_config, "_commit_hash", args.revision),
        "draft_config_sha256": hashlib.sha256(
            Path(args.draft_config).read_bytes()
        ).hexdigest(),
        "input_sha256": input_hash.hexdigest(),
        "target_kv_geometry": geometry,
        "state": "post_qk_norm_post_rope",
        "position_origin": 0,
        "padding": False,
        "max_length": args.max_length,
        "dtype": args.dtype,
    }
    (destination / "capture-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"samples": count, "format": manifest["format"]}))


if __name__ == "__main__":
    main()
