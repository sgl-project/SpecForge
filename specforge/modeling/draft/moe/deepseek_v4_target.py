# coding=utf-8
"""Read one DeepSeek-V4 target MoE layer as a warm-start source.

The DeepSeek-V4-Flash checkpoints store routed experts as packed FP4 e2m1
(two values per int8, low nibble first) with per-32 ue8m0 scales, the shared
expert and other linears as FP8 E4M3 with 128x128-block ue8m0 scales, the
gate in bf16 and the ``noaux_tc`` bias in fp32. This module dequantizes one
layer's ``ffn.*`` tensors to bf16 in the official naming
:func:`specforge.modeling.draft.moe.init.apply_warm_start` consumes
(``experts.{i}.w{1,2,3}.weight``, ``gate.weight``, ``gate.bias``,
``shared_experts.w{1,2,3}.weight``). Conventions follow the reference
``inference/convert.py``.

Layers below ``num_hash_layers`` route by token hash (``gate.tid2eid``) and
carry no learned gate; they are rejected as warm-start sources.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Mapping, Tuple

import torch

FP4_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)
FP8_BLOCK = 128
FP4_GROUP = 32


def dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP8 E4M3 ``[out, in]`` with e8m0 scale ``[out/128, in/128]`` -> bf16."""
    out_dim, in_dim = weight.shape
    if out_dim % FP8_BLOCK or in_dim % FP8_BLOCK:
        raise ValueError(f"fp8 weight {tuple(weight.shape)} is not 128-block aligned")
    w = weight.float().view(
        out_dim // FP8_BLOCK, FP8_BLOCK, in_dim // FP8_BLOCK, FP8_BLOCK
    )
    w = w * scale.float()[:, None, :, None]
    return w.view(out_dim, in_dim).to(torch.bfloat16)


def dequant_fp4_packed(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Packed FP4 int8 ``[out, in/2]`` with e8m0 scale ``[out, in/32]`` -> bf16.

    Low nibble = even element, high nibble = odd element."""
    if weight.dtype != torch.int8:
        raise TypeError(f"packed fp4 weight must be int8, got {weight.dtype}")
    out_dim, half_in = weight.shape
    in_dim = half_in * 2
    x = weight.view(torch.uint8)
    decoded = torch.stack(
        [FP4_TABLE[(x & 0x0F).long()], FP4_TABLE[((x >> 4) & 0x0F).long()]], dim=-1
    ).view(out_dim, in_dim // FP4_GROUP, FP4_GROUP)
    decoded = decoded * scale.float()[:, :, None]
    return decoded.view(out_dim, in_dim).to(torch.bfloat16)


def dequantize_ffn_tensors(
    raw: Mapping[str, torch.Tensor], prefix: str
) -> Dict[str, torch.Tensor]:
    """``{prefix}...`` raw tensors of one MoE layer -> official-relative bf16 dict."""
    out: Dict[str, torch.Tensor] = {}
    for name, tensor in raw.items():
        if not name.startswith(prefix) or name.endswith(".scale"):
            continue
        rel = name[len(prefix) :]
        if tensor.dtype == torch.float8_e4m3fn:
            value = dequant_fp8_block(tensor, raw[name[: -len(".weight")] + ".scale"])
        elif tensor.dtype == torch.int8:
            value = dequant_fp4_packed(tensor, raw[name[: -len(".weight")] + ".scale"])
        elif rel == "gate.bias":
            value = tensor.float()
        elif rel == "gate.tid2eid":
            raise ValueError(f"{prefix} is a hash-routed layer (no learned gate)")
        else:
            value = tensor.to(torch.bfloat16)
        out[rel] = value
    if "gate.bias" not in out or "gate.weight" not in out:
        raise ValueError(
            f"{prefix} has no learned gate; pick a layer >= num_hash_layers"
        )
    return out


def _iter_layer_tensors(
    snapshot_dir: str, prefix: str
) -> Iterable[Tuple[str, torch.Tensor]]:
    from safetensors.torch import safe_open

    index = json.load(open(os.path.join(snapshot_dir, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    shards = sorted({v for k, v in weight_map.items() if k.startswith(prefix)})
    if not shards:
        raise KeyError(f"no tensors with prefix {prefix!r} in {snapshot_dir}")
    for shard in shards:
        with safe_open(
            os.path.join(snapshot_dir, shard), framework="pt", device="cpu"
        ) as h:
            for name in h.keys():
                if name.startswith(prefix):
                    yield name, h.get_tensor(name)


def load_target_moe_layer(snapshot_dir: str, layer_id: int) -> Dict[str, torch.Tensor]:
    """Dequantized ``layers.{layer_id}.ffn.*`` of a DeepSeek-V4 checkpoint dir."""
    prefix = f"layers.{layer_id}.ffn."
    return dequantize_ffn_tensors(
        dict(_iter_layer_tensors(snapshot_dir, prefix)), prefix
    )
