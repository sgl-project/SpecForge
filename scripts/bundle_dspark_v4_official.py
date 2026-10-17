#!/usr/bin/env python3
"""Bundle a trained DSparkV4 drafter into the official checkpoint layout.

Takes a SpecForge-trained DSparkV4DraftModel (an ``specforge export --to hf``
directory, or a training checkpoint dir/``training_state.pt``) and produces a
directory laid out exactly like the official
``deepseek-ai/DeepSeek-V4-Flash-0731`` release:

  - the target shards, config.json, tokenizer files, index and the encoding/
    dir are HARD-LINKED from the official snapshot (config is byte-identical
    by construction: its dspark_* fields describe our drafter too, which this
    script checks);
  - the shards holding ``mtp.*`` are rewritten with our trained tensors
    quantized to the official dtypes: FP8 E4M3 + 128x128 ue8m0 block scales
    for attention/shared-expert/main_proj, packed FP4 e2m1 + per-32 ue8m0
    scales for routed experts, bf16/fp32 for the rest.

The result serves with the exact official command:
  sglang serve <bundle> --tp 4 --speculative-algorithm DSPARK ...

Self-test (run BEFORE trusting any bundle): dequantize the official mtp
weights, requantize with this script's quantizers, and verify the dequantized
round trip matches the official values exactly:
  python3 scripts/bundle_dspark_v4_official.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
from typing import Dict

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from dspark_v4_official_weights import (  # noqa: E402
    DEFAULT_SNAPSHOT,
    iter_official_tensors,
)

from specforge.modeling.draft.moe.families.deepseek import (  # noqa: E402
    dequantize_fp4,
    dequantize_fp8,
    quantize_fp4,
    quantize_fp8,
    read_config,
)


def _official_headers(snapshot: str) -> Dict[str, dict]:
    index = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    shards = sorted({v for k, v in weight_map.items() if k.startswith("mtp.")})
    headers: Dict[str, dict] = {}
    for shard in shards:
        path = os.path.join(snapshot, shard)
        with open(path, "rb") as stream:
            n = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(n))
        for key, meta in header.items():
            if key.startswith("mtp."):
                headers[key] = dict(meta, shard=shard)
    return headers


def quantize_state(
    state: Dict[str, torch.Tensor], headers: Dict[str, dict]
) -> Dict[str, torch.Tensor]:
    """Quantize an official-named mtp state dict to the official dtypes."""
    out: Dict[str, torch.Tensor] = {}
    for key, meta in headers.items():
        if key.endswith(".scale"):
            continue
        dtype = meta["dtype"]
        base = key[: -len(".weight")] if key.endswith(".weight") else key
        tensor = state[key]
        if dtype == "F8_E4M3":
            out[key], out[base + ".scale"] = quantize_fp8(tensor)
        elif dtype == "I8":
            out[key], out[base + ".scale"] = quantize_fp4(tensor)
        elif dtype == "F32":
            out[key] = tensor.float()
        elif dtype == "BF16":
            out[key] = tensor.to(torch.bfloat16)
        else:
            raise ValueError(f"unexpected official dtype {dtype} for {key}")
        assert tuple(out[key].shape) == tuple(meta["shape"]), (
            key,
            tuple(out[key].shape),
            meta["shape"],
        )
    return out


def verify_against_official(bundle: Dict[str, torch.Tensor], headers) -> None:
    dtype_names = {
        torch.float8_e4m3fn: "F8_E4M3",
        torch.float8_e8m0fnu: "F8_E8M0",
        torch.int8: "I8",
        torch.bfloat16: "BF16",
        torch.float32: "F32",
    }
    assert set(bundle) == set(headers), (
        sorted(set(headers) - set(bundle))[:5],
        sorted(set(bundle) - set(headers))[:5],
    )
    for key, meta in headers.items():
        got = bundle[key]
        assert dtype_names[got.dtype] == meta["dtype"], (key, got.dtype, meta["dtype"])
        assert tuple(got.shape) == tuple(meta["shape"]), (key, got.shape, meta["shape"])


def load_draft_state(source: str) -> Dict[str, torch.Tensor]:
    """mtp.* bf16 state, in the checkpoint (official) naming, from an HF export
    dir or a training checkpoint."""
    if os.path.isdir(source):
        candidates = [
            name for name in os.listdir(source) if name.endswith(".safetensors")
        ]
        if candidates:
            from safetensors.torch import safe_open

            state = {}
            for name in candidates:
                with safe_open(
                    os.path.join(source, name), framework="pt", device="cpu"
                ) as handle:
                    for key in handle.keys():
                        state[key] = handle.get_tensor(key)
            return {
                k: v
                for k, v in state.items()
                if k.startswith(("mtp.", "markov_head.", "confidence_head."))
            }
    from specforge.export.checkpoint_io import resolve_training_state

    state = resolve_training_state(source)
    return dict(state["draft_state_dict"])


def _official_head_names(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Root-owned head names (module tree) -> the official ``mtp.<last>.*``
    names."""
    stages = {int(k.split(".")[1]) for k in state if k.startswith("mtp.")}
    last = f"mtp.{max(stages)}."
    out = {}
    for key, value in state.items():
        if key.startswith(("markov_head.", "confidence_head.")):
            key = last + key
        out[key] = value
    return out


def _check_expert_counts(draft_source: str, snapshot: str) -> None:
    """The bundle reuses the official config.json, so the drafter's routing
    counts must be the ones it declares."""
    path = os.path.join(draft_source, "config.json")
    if not os.path.isfile(path):
        return
    with open(path) as stream:
        draft = json.load(stream)
    official = read_config(snapshot)
    for draft_key, official_key in (
        ("n_routed_experts", "dspark_n_routed_experts"),
        ("num_experts_per_tok", "dspark_num_experts_per_tok"),
    ):
        fallback = official_key[len("dspark_") :]
        expected = official.get(official_key) or official[fallback]
        if draft_key in draft and int(draft[draft_key]) != int(expected):
            raise SystemExit(
                f"drafter {draft_key}={draft[draft_key]} but the official config "
                f"serves {expected}; the bundle would route differently"
            )


def self_test(snapshot: str) -> None:
    print("quantizer round-trip self-test on official weights ...")
    headers = _official_headers(snapshot)
    raw = dict(iter_official_tensors(snapshot))
    checked = 0
    for key, meta in headers.items():
        if key.endswith(".scale"):
            continue
        base = key[: -len(".weight")] if key.endswith(".weight") else key
        tensor = raw[key]
        if meta["dtype"] == "F8_E4M3":
            deq = dequantize_fp8(tensor, raw[base + ".scale"])
            rt = dequantize_fp8(*quantize_fp8(deq))
        elif meta["dtype"] == "I8":
            deq = dequantize_fp4(tensor, raw[base + ".scale"])
            rt = dequantize_fp4(*quantize_fp4(deq))
        else:
            continue
        if not torch.equal(rt, deq):
            diff = (rt.float() - deq.float()).abs()
            raise AssertionError(
                f"{key}: round-trip mismatch, max abs diff {diff.max().item()}"
            )
        checked += 1
    print(f"self-test PASS: {checked} quantized tensors round-trip exactly")


_LINK_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
)


def bundle(draft_source: str, output_dir: str, snapshot: str) -> None:
    headers = _official_headers(snapshot)
    if os.path.isdir(draft_source):
        _check_expert_counts(draft_source, snapshot)
    print(f"loading trained drafter state from {draft_source} ...")
    state = _official_head_names(load_draft_state(draft_source))
    expected_keys = {k for k in headers if not k.endswith(".scale")}
    missing = sorted(expected_keys - set(state))
    extra = sorted(set(state) - expected_keys)
    assert not missing, f"drafter state missing tensors: {missing[:8]}"
    assert not extra, f"drafter state has unexpected tensors: {extra[:8]}"

    print("quantizing to official layout ...")
    quantized = quantize_state(state, headers)
    verify_against_official(quantized, headers)

    os.makedirs(output_dir, exist_ok=True)
    index = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    mtp_shards = sorted({headers[k]["shard"] for k in headers})
    target_shards = sorted(
        {v for k, v in weight_map.items() if k not in headers} - set(mtp_shards)
    )
    # Shards that mix target and mtp tensors would need rewriting; the official
    # layout keeps them disjoint (the trailing shards are mtp-only), assert
    # that holds.
    mixed = {weight_map[k] for k in weight_map if k not in headers} & set(mtp_shards)
    assert not mixed, f"target tensors share mtp shards: {sorted(mixed)}"

    def _link(src: str, dst: str) -> None:
        # HF snapshots are symlink farms into blobs/; link the real file so
        # the bundle stands alone (a linked relative symlink would dangle).
        if os.path.lexists(dst):
            os.remove(dst)
        os.link(os.path.realpath(src), dst)

    for name in target_shards:
        _link(os.path.join(snapshot, name), os.path.join(output_dir, name))
    for name in _LINK_FILES:
        src = os.path.join(snapshot, name)
        if os.path.exists(src):
            _link(src, os.path.join(output_dir, name))
    enc_src = os.path.join(snapshot, "encoding")
    enc_dst = os.path.join(output_dir, "encoding")
    if os.path.isdir(enc_src) and not os.path.exists(enc_dst):
        shutil.copytree(enc_src, enc_dst)

    from safetensors.torch import save_file

    for shard in mtp_shards:
        tensors = {
            k: quantized[k].contiguous()
            for k in sorted(headers)
            if headers[k]["shard"] == shard
        }
        out_path = os.path.join(output_dir, shard)
        print(f"writing {shard} ({len(tensors)} tensors) ...")
        save_file(tensors, out_path)

    # Final verification: identical key set + dtype/shape per tensor.
    got_headers = _official_headers(output_dir)
    for key, meta in headers.items():
        got = got_headers[key]
        assert got["dtype"] == meta["dtype"], (key, got["dtype"], meta["dtype"])
        assert got["shape"] == meta["shape"], (key, got["shape"], meta["shape"])
    import filecmp

    assert filecmp.cmp(
        os.path.join(output_dir, "config.json"),
        os.path.join(snapshot, "config.json"),
        shallow=False,
    ), "config.json differs from official"
    print(f"bundle complete: {output_dir}")
    print("serve with:")
    print(
        f"  sglang serve {output_dir} --trust-remote-code --tp 4 "
        "--speculative-algorithm DSPARK --moe-runner-backend marlin "
        "--mem-fraction-static 0.90 --chunked-prefill-size 4096"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", help="HF export dir or training checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--official-snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(args.official_snapshot)
        return
    if not args.draft or not args.output_dir:
        parser.error("--draft and --output-dir are required (or use --self-test)")
    bundle(args.draft, args.output_dir, args.official_snapshot)


if __name__ == "__main__":
    main()
