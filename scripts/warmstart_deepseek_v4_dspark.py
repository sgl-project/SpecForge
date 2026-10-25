#!/usr/bin/env python3
"""Import native DeepSeek V4 Flash/Pro DSpark weights into SpecForge.

All tensors (including projection, mHC and heads) must match; no slicing or
random fallback is permitted. FP4 E2M1 experts and block FP8 projections are
dequantized on CPU, one tensor at a time, into bounded output shards.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def dequantize_weight(weight, scale, expected_shape):
    """Decode the actual safetensors dtypes, not a guessed byte exponent."""
    expected_shape = tuple(expected_shape)
    if weight.dtype in (torch.int8, torch.uint8):
        if weight.ndim != 2 or expected_shape != (weight.shape[0], weight.shape[1] * 2):
            raise ValueError("packed FP4 shape does not match expected dense shape")
        if scale is None or tuple(scale.shape) != (
            expected_shape[0],
            math.ceil(expected_shape[1] / 32),
        ):
            raise ValueError(
                "FP4 requires one E8M0 scale per row and 32 input channels"
            )
        if scale.dtype != torch.float8_e8m0fnu:
            raise ValueError("V4 FP4 scales must be typed E8M0, not raw exponent bytes")
        table = weight.new_tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            dtype=torch.float32,
        )
        packed = weight.view(torch.uint8)
        dense = torch.stack(
            (table[(packed & 15).long()], table[(packed >> 4).long()]), -1
        ).flatten(-2)
        # E8M0.float() already decodes the exponent. Do NOT subtract 127 again.
        return dense * scale.float().repeat_interleave(32, -1)[:, : expected_shape[1]]
    if weight.dtype == torch.float8_e4m3fn:
        if weight.ndim != 2 or tuple(weight.shape) != expected_shape:
            raise ValueError("FP8 projection shape mismatch")
        if scale is None or tuple(scale.shape) != tuple(
            math.ceil(n / 128) for n in expected_shape
        ):
            raise ValueError("FP8 requires a 128x128 block-scale grid")
        scales = scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
        return weight.float() * scales[: weight.shape[0], : weight.shape[1]]
    if scale is not None:
        raise ValueError(
            f"unexpected scale on unquantized tensor of dtype {weight.dtype}"
        )
    if not weight.is_floating_point() or tuple(weight.shape) != expected_shape:
        raise ValueError(
            f"unsupported dtype/shape: {weight.dtype}, {tuple(weight.shape)}, expected {expected_shape}"
        )
    return weight.float()


class CheckpointReader:
    def __init__(self, path):
        self.path = Path(path)
        self.weight_map = json.loads(
            (self.path / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.handles = {}

    def handle(self, key):
        shard = self.weight_map[key]
        if shard not in self.handles:
            self.handles[shard] = safe_open(
                str(self.path / shard), framework="pt", device="cpu"
            )
        return self.handles[shard]

    def scale_key(self, key):
        candidate = key.removesuffix("weight") + "scale"
        return (
            candidate
            if key.endswith(".weight") and candidate in self.weight_map
            else None
        )

    def validate(self, contract):
        missing = set(contract) - set(self.weight_map)
        unexpected = {
            k
            for k in self.weight_map
            if k.startswith("mtp.") and not k.endswith(".scale")
        } - set(contract)
        if missing or unexpected:
            raise ValueError(
                f"draft key mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        for key, shape in contract.items():
            view = self.handle(key).get_slice(key)
            stored = tuple(view.get_shape())
            dtype = view.get_dtype()
            dense = (
                (stored[0], stored[1] * 2)
                if dtype in {"I8", "U8"} and len(stored) == 2
                else stored
            )
            if dense != shape:
                raise ValueError(
                    f"{key}: checkpoint dense shape {dense} != draft {shape}"
                )
            sk = self.scale_key(key)
            if dtype in {"I8", "U8", "F8_E4M3"} and sk is None:
                raise ValueError(f"{key}: missing quantization scale")

    def get(self, key, shape):
        sk = self.scale_key(key)
        scale = self.handle(sk).get_tensor(sk) if sk else None
        return dequantize_weight(self.handle(key).get_tensor(key), scale, shape)


def build_contract(config_path):
    from specforge.modeling.draft.deepseek_v4_dspark import (
        DeepseekV4DSparkConfig,
        DeepseekV4DSparkDraftModel,
    )

    payload = json.loads(Path(config_path).read_text())
    if payload.get("architectures") != ["DeepseekV4DSparkDraftModel"]:
        raise ValueError("expected a DeepseekV4DSparkDraftModel config")
    config = DeepseekV4DSparkConfig.from_dict(payload)
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = DeepseekV4DSparkDraftModel(config)
    return config, {k: tuple(v.shape) for k, v in model.state_dict().items()}


def validate_source_config(config, source):
    if source.get("model_type") != "deepseek_v4":
        raise ValueError("expected a native DeepSeek V4 source checkpoint")
    blocks = (source.get("quantization_config") or {}).get(
        "weight_block_size", [128, 128]
    )
    if blocks != [128, 128]:
        raise ValueError("DeepSeek V4 FP8 projections require 128x128 blocks")
    for name in (
        "vocab_size",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "q_lora_rank",
        "qk_rope_head_dim",
        "o_groups",
        "o_lora_rank",
        "rope_theta",
        "sliding_window",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "routed_scaling_factor",
        "scoring_func",
        "norm_topk_prob",
        "swiglu_limit",
        "hc_mult",
        "hc_sinkhorn_iters",
        "hc_eps",
        "rms_norm_eps",
    ):
        if getattr(config, name) != source.get(name):
            raise ValueError(
                f"target/draft {name} mismatch: {source.get(name)!r} vs {getattr(config, name)!r}"
            )
    if config.num_target_layers != source["num_hidden_layers"]:
        raise ValueError("num_target_layers differs from the teacher")
    for source_key, draft_key in (
        ("dspark_target_layer_ids", "target_layer_ids"),
        ("dspark_noise_token_id", "mask_token_id"),
        ("dspark_markov_rank", "markov_rank"),
    ):
        if source[source_key] != config.dflash_config[draft_key]:
            raise ValueError(f"{source_key} differs from the pretrained draft")
    ratios = source["compress_ratios"][config.num_target_layers :]
    if ratios != [0] * config.num_hidden_layers:
        raise ValueError("expected the three uncompressed official DSpark stages")


def import_checkpoint(
    target_path, config_path, output, *, shard_size_bytes=2**30, dry_run=False
):
    if shard_size_bytes <= 0:
        raise ValueError("shard_size_bytes must be positive")
    config, contract = build_contract(config_path)
    source = json.loads((Path(target_path) / "config.json").read_text())
    validate_source_config(config, source)
    reader = CheckpointReader(target_path)
    reader.validate(contract)
    report = {
        "source": str(Path(target_path).resolve()),
        "source_block_size": source["dspark_block_size"],
        "block_size": config.block_size,
        "target_layer_ids": config.dflash_config["target_layer_ids"],
        "tensor_count": len(contract),
        "elements": sum(math.prod(s) for s in contract.values()),
        "randomly_initialized": [],
        "quantization": "dequantized BF16 weights; FP32 mHC/sink/router bias retained in import",
    }
    if dry_run:
        return report
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent)
    )
    weight_map, buffer = {}, {}
    buffered_bytes = total_bytes = shard_no = 0

    def flush():
        nonlocal buffer, buffered_bytes, shard_no
        if not buffer:
            return
        shard_no += 1
        name = f"model-{shard_no:05d}.safetensors"
        save_file(buffer, str(staging / name), metadata={"format": "pt"})
        weight_map.update({k: name for k in buffer})
        buffer, buffered_bytes = {}, 0

    try:
        for key, shape in contract.items():
            tensor = reader.get(key, shape)
            if not torch.isfinite(tensor).all():
                raise ValueError(f"non-finite imported tensor: {key}")
            keep_fp32 = ".hc_" in key or key.endswith((".attn_sink", ".gate.bias"))
            tensor = tensor.to(
                torch.float32 if keep_fp32 else torch.bfloat16
            ).contiguous()
            size = tensor.numel() * tensor.element_size()
            if buffered_bytes + size > shard_size_bytes:
                flush()
            buffer[key] = tensor
            buffered_bytes += size
            total_bytes += size
        flush()
        (staging / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
                indent=2,
            )
            + "\n"
        )
        config.save_pretrained(staging)
        (staging / "warmstart_manifest.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-path", required=True)
    p.add_argument(
        "--draft-config", default="configs/deepseek-v4-flash-0731-dspark.json"
    )
    p.add_argument("--output", required=True)
    p.add_argument("--shard-size-gib", type=float, default=1)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.shard_size_gib <= 0:
        p.error("--shard-size-gib must be positive")
    torch.set_num_threads(8)
    print(
        json.dumps(
            import_checkpoint(
                args.target_path,
                args.draft_config,
                args.output,
                shard_size_bytes=int(args.shard_size_gib * 2**30),
                dry_run=args.dry_run,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
