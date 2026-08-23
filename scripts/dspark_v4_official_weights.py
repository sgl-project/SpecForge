#!/usr/bin/env python3
"""Load / dequantize the official DeepSeek-V4-Flash DSpark ``mtp.*`` weights.

The official checkpoint stores the drafter quantized: attention/shared-expert/
main_proj linears as FP8 E4M3 with 128x128-block ue8m0 scales, routed experts
as packed FP4 e2m1 (two values per int8, low nibble first) with per-32 ue8m0
scales. This module dequantizes them to bf16 with the exact conventions of the
reference ``inference/convert.py``, producing a state dict loadable into
``DSparkV4DraftModel``.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Dict, Iterable, Optional

import torch

# The mtp.* drafter bundled with deepseek-ai/DeepSeek-V4-Flash-0731. NOTE:
# the standalone DeepSeek-V4-Flash-DSpark repo is a DIFFERENT model release
# (both its target shards and its drafter weights differ from 0731, despite a
# byte-identical config.json) — do not mix the two.
DEFAULT_SNAPSHOT = (
    "/cluster-storage/models/models--deepseek-ai--DeepSeek-V4-Flash-0731/"
    "snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062"
)

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

FP8_BLOCK = 128
FP4_GROUP = 32


def dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP8 E4M3 weight [out, in] with e8m0 scale [ceil(out/128), ceil(in/128)]."""
    out_dim, in_dim = weight.shape
    assert out_dim % FP8_BLOCK == 0 and in_dim % FP8_BLOCK == 0, weight.shape
    w = weight.float().view(
        out_dim // FP8_BLOCK, FP8_BLOCK, in_dim // FP8_BLOCK, FP8_BLOCK
    )
    w = w * scale.float()[:, None, :, None]
    return w.view(out_dim, in_dim).to(torch.bfloat16)


def dequant_fp4_packed(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Packed FP4 int8 weight [out, in//2] with e8m0 scale [out, in//32].

    Element order matches the reference decode: for each byte, the low nibble
    is the even element and the high nibble the odd element.
    """
    assert weight.dtype == torch.int8
    out_dim, half_in = weight.shape
    in_dim = half_in * 2
    x = weight.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    decoded = torch.stack(
        [FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1
    ).view(out_dim, in_dim)
    decoded = decoded.view(out_dim, in_dim // FP4_GROUP, FP4_GROUP)
    decoded = decoded * scale.float()[:, :, None]
    return decoded.view(out_dim, in_dim).to(torch.bfloat16)


def _read_safetensors_header(path: str) -> dict:
    with open(path, "rb") as stream:
        header_len = struct.unpack("<Q", stream.read(8))[0]
        return json.loads(stream.read(header_len))


def iter_official_tensors(
    snapshot_dir: str, prefix: str = "mtp."
) -> Iterable[tuple[str, torch.Tensor]]:
    from safetensors.torch import safe_open

    index = json.load(
        open(os.path.join(snapshot_dir, "model.safetensors.index.json"))
    )
    weight_map: Dict[str, str] = index["weight_map"]
    shards = sorted({v for k, v in weight_map.items() if k.startswith(prefix)})
    for shard in shards:
        with safe_open(
            os.path.join(snapshot_dir, shard), framework="pt", device="cpu"
        ) as handle:
            for name in handle.keys():
                if name.startswith(prefix):
                    yield name, handle.get_tensor(name)


def load_official_mtp_state(
    snapshot_dir: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, torch.Tensor]:
    """Dequantized ``mtp.*`` state dict in the official tensor naming."""
    snapshot_dir = snapshot_dir or os.environ.get(
        "DSPARK_OFFICIAL_SNAPSHOT", DEFAULT_SNAPSHOT
    )
    raw: Dict[str, torch.Tensor] = dict(iter_official_tensors(snapshot_dir))
    state: Dict[str, torch.Tensor] = {}
    for name, tensor in raw.items():
        if name.endswith(".scale"):
            continue
        if tensor.dtype == torch.float8_e4m3fn:
            scale = raw[name[: -len(".weight")] + ".scale"]
            value = dequant_fp8_block(tensor, scale)
        elif tensor.dtype == torch.int8:
            scale = raw[name[: -len(".weight")] + ".scale"]
            value = dequant_fp4_packed(tensor, scale)
        else:
            value = tensor
        state[name] = value.to(dtype) if value.is_floating_point() else value
    return state


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    state = load_official_mtp_state(args.snapshot)
    total = sum(v.numel() for v in state.values())
    print(f"{len(state)} tensors, {total/1e9:.2f}B params")
    if args.summary:
        for name in sorted(state)[:40]:
            print(name, tuple(state[name].shape), state[name].dtype)
