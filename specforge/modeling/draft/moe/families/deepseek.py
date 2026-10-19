# coding=utf-8
"""DeepSeek-V4 MoE family: DeepSeek-V4-Flash(-0731) and the DSpark drafter
bundled with it.

**Preset ``deepseek_v4``**: ``sqrt(softplus)`` scores; top-k on the scores
plus an aux-loss-free selection bias (``noaux_tc``: after every step the sign
rule moves each expert's bias against its load); the combine weights are the
*unbiased* scores of the chosen experts, renormalized and scaled by 1.5; one
ungated SwiGLU shared expert as wide as a routed expert; SwiGLU clamped at 10
(``up`` on both sides, ``gate`` from above). The targets set
``n_group == topk_group``, so group-limited routing is off.

**Layout ``deepseek``**: the official checkpoint naming, which DeepSeek's
``inference/convert.py`` and SGLang's DSpark loader read::

    gate.weight                     router [E, hidden]
    gate.bias                       selection bias, fp32   <- gate.balance.bias
    experts.{i}.w1|w2|w3.weight     <- experts.w1|w2|w3 [E, out, in]
    shared_experts.w1|w2|w3.weight  (module names)

Hash-routed target layers (``gate.tid2eid``) have no learned router and are
rejected.
``gate.bias`` only steers selection here, so router centering, which an
export folds into a router *logit* bias, has no name in this layout.

**Serving**: a DSpark drafter rides inside the target checkpoint as
``mtp.*``. SGLang's ``DeepseekV4ForCausalLMDSpark`` builds its MoE from the
target's routing config and overrides only the expert counts, from
``dspark_n_routed_experts`` and ``dspark_num_experts_per_tok``.

**Checkpoint dtypes**: routed experts are packed FP4 e2m1 (two values per
byte, low nibble first) with one ue8m0 scale per 32 inputs; the other large
linears are FP8 E4M3 with one ue8m0 scale per 128x128 block. The
(de)quantizers below follow ``inference/convert.py``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterator, Mapping, Sequence, Tuple

import torch

from ..checkpoint import register_layout
from ..config import MoEConfig, register_moe_preset

register_moe_preset(
    "deepseek_v4",
    scoring_func="sqrtsoftplus",
    norm_topk_prob=True,
    routed_scaling_factor=1.5,
    n_shared_experts=1,
    swiglu_limit=10.0,
    router="topk",
    balance="noaux_tc",
    shared_expert_gate="none",
    layout="deepseek",
)


# -- layout ------------------------------------------------------------------------

_EXPERT = re.compile(r"^experts\.(\d+)\.(w[123])\.weight$")
#: Tensors with one row per routed expert besides the experts themselves.
_ROUTER_ROWS = ("gate.weight", "gate.bias")


def to_deepseek_file(layer: Dict[str, Any]) -> Dict[str, Any]:
    """One MoE layer, module naming -> DeepSeek file naming."""
    if "gate.bias" in layer:
        raise ValueError(
            "the deepseek layout's gate.bias is the selection-only bias, so it "
            "cannot hold a router logit bias (moe_router_bias, or the one "
            "moe_router_center folds into at export); train without them or "
            "export with moe_layout='native'"
        )
    out: Dict[str, Any] = {}
    for key, value in layer.items():
        if key in ("experts.w1", "experts.w2", "experts.w3"):
            name = key[len("experts.") :]
            for i in range(value.shape[0]):
                out[f"experts.{i}.{name}.weight"] = value[i]
        elif key == "gate.balance.bias":
            out["gate.bias"] = value
        else:
            out[key] = value
    return out


def from_deepseek_file(layer: Dict[str, Any]) -> Dict[str, Any]:
    """One MoE layer, DeepSeek file naming -> module naming."""
    groups: Dict[str, Dict[int, torch.Tensor]] = {}
    out: Dict[str, Any] = {}
    for key, value in layer.items():
        if m := _EXPERT.match(key):
            groups.setdefault(m[2], {})[int(m[1])] = value
        elif key == "gate.bias":
            out["gate.balance.bias"] = value
        elif key == "gate.tid2eid":
            raise ValueError(
                "hash-routed MoE layer (gate.tid2eid): its experts are picked by "
                "token id, not by a learned router; use a learned-routing layer"
            )
        else:
            out[key] = value
    for name, members in groups.items():
        n = max(members) + 1
        if sorted(members) != list(range(n)):
            raise KeyError(
                f"experts.*.{name}.weight is missing expert indices: "
                f"have {sorted(members)}"
            )
        out[f"experts.{name}"] = torch.stack([members[i] for i in range(n)], dim=0)
    return out


def select_deepseek_experts(
    layer: Dict[str, Any], ids: Sequence[int]
) -> Dict[str, Any]:
    """Keep experts ``ids`` (renumbered ``0..len(ids)-1``) with their router rows."""
    rows = torch.as_tensor(list(ids), dtype=torch.long)
    out: Dict[str, Any] = {}
    for new, old in enumerate(ids):
        for w in ("w1", "w2", "w3"):
            out[f"experts.{new}.{w}.weight"] = layer[f"experts.{old}.{w}.weight"]
    for key, value in layer.items():
        if key in _ROUTER_ROWS:
            out[key] = value.index_select(0, rows)
        elif key.startswith("shared_experts."):
            out[key] = value
    return out


#: The routing SGLang's DeepseekV4ForCausalLMDSpark takes from the target
#: config; a drafter may differ from its target only in the expert counts.
TARGET_ROUTING = {
    "scoring_func": "sqrtsoftplus",
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.5,
    "swiglu_limit": 10.0,
    "n_shared_experts": 1,
    "shared_expert_gate": "none",
    "balance": "noaux_tc",
}


def deepseek_serving_config(cfg: MoEConfig) -> Dict[str, Any]:
    """``config.json`` fields for ``DeepseekV4ForCausalLMDSpark``.

    Refuses recipes that class cannot run, so an export never silently serves
    a different routing than the one trained.
    """
    problems = [
        f"{name}={getattr(cfg, name)!r} (DeepSeek-V4 uses {value!r})"
        for name, value in TARGET_ROUTING.items()
        if getattr(cfg, name) != value
    ]
    if cfg.group_limited:
        problems.append("group-limited routing")
    if cfg.shared_expert_intermediate_size != cfg.moe_intermediate_size:
        problems.append("a shared expert wider than the routed experts")
    if cfg.router_center != "none":
        problems.append(f"moe_router_center={cfg.router_center!r}")
    if cfg.router_bias:
        problems.append("a router logit bias")
    if problems:
        raise ValueError(
            "DeepseekV4ForCausalLMDSpark cannot serve this MoE recipe: "
            + ", ".join(problems)
            + "; export with moe_layout='native' for SpecForge-only evaluation"
        )
    return {
        "dspark_n_routed_experts": cfg.n_routed_experts,
        "dspark_num_experts_per_tok": cfg.num_experts_per_tok,
    }


register_layout(
    "deepseek",
    to_file=to_deepseek_file,
    from_file=from_deepseek_file,
    serving_config=deepseek_serving_config,
)


# -- checkpoint dtypes -------------------------------------------------------------

#: Routed-expert FP4 scale group along the input dimension.
FP4_GROUP = 32
#: FP8 weight scale block (``quantization_config.weight_block_size``).
FP8_BLOCK = 128
#: e2m1 code -> value; bit 3 is the sign (``inference/convert.py``).
FP4_E2M1_VALUES = (
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
)
_FP4_MAGNITUDE_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_FP8_MAX = 448.0
_FP4_MAX = 6.0


def _ue8m0(amax: torch.Tensor, qmax: float) -> torch.Tensor:
    """The smallest power of two ``s`` with ``amax / s <= qmax``."""
    return torch.exp2(torch.ceil(torch.log2(amax / qmax)))


def dequantize_fp8(
    weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """E4M3 ``[out, in]`` with one scale per 128x128 tile (``scale`` is
    ``[ceil(out/128), ceil(in/128)]``) -> ``[out, in]``."""
    block = FP8_BLOCK
    out_dim, in_dim = weight.shape
    expected = (-(-out_dim // block), -(-in_dim // block))
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"fp8 weight {tuple(weight.shape)} needs a {expected} scale, "
            f"got {tuple(scale.shape)}"
        )
    tiles = scale.float().repeat_interleave(block, 0)[:out_dim]
    tiles = tiles.repeat_interleave(block, 1)[:, :in_dim]
    return (weight.float() * tiles).to(dtype)


def quantize_fp8(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``[out, in]`` -> (E4M3 weight, ue8m0 scale per 128x128 tile)."""
    block = FP8_BLOCK
    out_dim, in_dim = weight.shape
    if out_dim % block or in_dim % block:
        raise ValueError(f"{tuple(weight.shape)} is not {block}-block aligned")
    w = weight.float().view(out_dim // block, block, in_dim // block, block)
    scale = _ue8m0(w.abs().amax(dim=(1, 3)).clamp_min(1e-30), _FP8_MAX)
    q = (w / scale[:, None, :, None]).clamp(-_FP8_MAX, _FP8_MAX)
    return q.view(out_dim, in_dim).to(torch.float8_e4m3fn), scale.to(
        torch.float8_e8m0fnu
    )


def dequantize_fp4(
    weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Packed e2m1 ``[out, in/2]`` (int8; low nibble = even element) with one
    scale per 32 inputs (``[out, in/32]``) -> ``[out, in]``."""
    if weight.dtype not in (torch.int8, torch.uint8):
        raise TypeError(f"packed fp4 weight must be int8, got {weight.dtype}")
    out_dim, half_in = weight.shape
    in_dim = 2 * half_in
    if tuple(scale.shape) != (out_dim, in_dim // FP4_GROUP):
        raise ValueError(
            f"fp4 weight {tuple(weight.shape)} needs a "
            f"{(out_dim, in_dim // FP4_GROUP)} scale, got {tuple(scale.shape)}"
        )
    table = torch.tensor(FP4_E2M1_VALUES, dtype=torch.float32, device=weight.device)
    codes = weight.view(torch.uint8)
    values = torch.stack(
        [table[(codes & 0x0F).long()], table[(codes >> 4).long()]], dim=-1
    ).view(out_dim, in_dim // FP4_GROUP, FP4_GROUP)
    return (values * scale.float().unsqueeze(-1)).view(out_dim, in_dim).to(dtype)


def quantize_fp4(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``[out, in]`` -> (packed e2m1 int8 ``[out, in/2]``, ue8m0 scale ``[out, in/32]``).

    Values snap to the nearest e2m1 magnitude (ties toward the smaller one);
    values already on the grid, like dequantized official weights, round-trip
    exactly.
    """
    out_dim, in_dim = weight.shape
    if in_dim % FP4_GROUP:
        raise ValueError(f"{tuple(weight.shape)} is not {FP4_GROUP}-group aligned")
    w = weight.float().view(out_dim, in_dim // FP4_GROUP, FP4_GROUP)
    scale = _ue8m0(w.abs().amax(dim=-1).clamp_min(_FP4_MAX * 2.0**-126), _FP4_MAX)
    q = (w / scale.unsqueeze(-1)).clamp(-_FP4_MAX, _FP4_MAX).view(out_dim, in_dim)
    midpoints = torch.tensor(_FP4_MAGNITUDE_MIDPOINTS, device=weight.device)
    code = torch.bucketize(q.abs(), midpoints) + torch.where(q < 0, 8, 0)
    nibbles = code.to(torch.uint8).view(out_dim, in_dim // 2, 2)
    packed = (nibbles[..., 0] | (nibbles[..., 1] << 4)).view(torch.int8)
    return packed, scale.to(torch.float8_e8m0fnu)


def dequantize_tensors(
    raw: Mapping[str, torch.Tensor], dtype: torch.dtype = torch.bfloat16
) -> Dict[str, torch.Tensor]:
    """Official tensors (``X.weight`` plus its ``X.scale``) -> dequantized
    ``dtype`` tensors without the scales; fp32 tensors (selection biases,
    hyper-connection parameters, attention sinks) keep fp32."""
    out: Dict[str, torch.Tensor] = {}
    for name, tensor in raw.items():
        if name.endswith(".scale"):
            continue
        if tensor.dtype in (torch.float8_e4m3fn, torch.int8):
            scale = raw[name[: -len(".weight")] + ".scale"]
            if tensor.dtype == torch.int8:
                out[name] = dequantize_fp4(tensor, scale, dtype)
            else:
                out[name] = dequantize_fp8(tensor, scale, dtype)
        elif tensor.dtype == torch.float32 or not tensor.is_floating_point():
            out[name] = tensor
        else:
            out[name] = tensor.to(dtype)
    return out


# -- target checkpoints -------------------------------------------------------------


def read_config(snapshot_dir: str) -> Dict[str, Any]:
    with open(os.path.join(snapshot_dir, "config.json"), encoding="utf-8") as stream:
        return json.load(stream)


def iter_checkpoint_tensors(
    snapshot_dir: str, prefix: str
) -> Iterator[Tuple[str, torch.Tensor]]:
    """``(name, tensor)`` for every checkpoint tensor whose name starts with
    ``prefix``, reading only the shards that hold them."""
    from safetensors import safe_open

    index_path = os.path.join(snapshot_dir, "model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as stream:
        weight_map = json.load(stream)["weight_map"]
    shards = sorted(
        {shard for name, shard in weight_map.items() if name.startswith(prefix)}
    )
    if not shards:
        raise KeyError(f"no tensors named {prefix}* in {snapshot_dir}")
    for shard in shards:
        path = os.path.join(snapshot_dir, shard)
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name.startswith(prefix):
                    yield name, handle.get_tensor(name)
