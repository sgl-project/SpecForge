# coding=utf-8
"""Convert DeepSeek-V4's DSpark draft between its ``mtp.*`` checkpoint and SpecForge.

The official checkpoint (``deepseek-ai/DeepSeek-V4-Flash-DSpark``) keeps the
draft's stages under ``mtp.{stage}.`` with the inference code's names: FP8
linears with one e8m0 ``.scale`` per 128x128 block, FP4 experts with one e8m0
``.scale`` per 32 input channels, fp32 hyper-connection and sink tensors.
SpecForge's ``DSparkDraftModel`` with ``attention_mode`` deepseek_v4 holds the
same tensors under transformers' names, unquantized, with the routed experts
fused into ``gate_up_proj`` / ``down_proj``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Optional, Tuple

import torch

OFFICIAL_PREFIX = "mtp."
FP8_BLOCK = 128
FP4_GROUP = 32
# E2M1 codes 0-15. Two codes per byte along the input, low nibble first (the
# packing SGLang's dequantize_nvfp4 documents and the official fp4 kernels use).
_FP4_VALUES = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
)

# Official stage tensor -> SpecForge tensor under ``layers.{stage}.``.
_LAYER_NAMES = {
    "attn.wq_a.weight": "self_attn.q_a_proj.weight",
    "attn.q_norm.weight": "self_attn.q_a_norm.weight",
    "attn.wq_b.weight": "self_attn.q_b_proj.weight",
    "attn.wkv.weight": "self_attn.kv_proj.weight",
    "attn.kv_norm.weight": "self_attn.kv_norm.weight",
    "attn.wo_a.weight": "self_attn.o_a_proj.weight",
    "attn.wo_b.weight": "self_attn.o_b_proj.weight",
    "attn.attn_sink": "self_attn.sinks",
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn.gate.weight": "mlp.gate.weight",
    "ffn.gate.bias": "mlp.gate.expert_bias",
    "ffn.shared_experts.w1.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn.shared_experts.w3.weight": "mlp.shared_expert.up_proj.weight",
    "ffn.shared_experts.w2.weight": "mlp.shared_expert.down_proj.weight",
    "hc_attn_fn": "attention_residual.mix_weight",
    "hc_attn_scale": "attention_residual.mix_scale",
    "hc_attn_base": "attention_residual.mix_bias",
    "hc_ffn_fn": "ffn_residual.mix_weight",
    "hc_ffn_scale": "ffn_residual.mix_scale",
    "hc_ffn_base": "ffn_residual.mix_bias",
}
# Tensors of the first and of the last stage that live on the SpecForge model.
_FIRST_STAGE_NAMES = {
    "main_proj.weight": "fc.weight",
    "main_norm.weight": "hidden_norm.weight",
}
_LAST_STAGE_NAMES = {
    "hc_head_fn": "residual_readout.mix_weight",
    "hc_head_scale": "residual_readout.mix_scale",
    "hc_head_base": "residual_readout.mix_bias",
    "norm.weight": "norm.weight",
    "markov_head.markov_w1.weight": "markov_head.markov_w1.weight",
    "markov_head.markov_w2.weight": "markov_head.markov_w2.weight",
    "confidence_head.proj.weight": "confidence_head.proj.weight",
}
_EXPERT = re.compile(r"ffn\.experts\.(\d+)\.(w1|w2|w3)\.weight")
_LAYER = re.compile(r"layers\.(\d+)\.(.*)")
# The official checkpoint keeps these in fp32; everything else is bf16.
_FP32_OFFICIAL = ("attn.attn_sink", "ffn.gate.bias", "hc_")


def dequantize_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """DeepSeek's FP8 layout: e4m3 ``[out, in]`` with one e8m0 scale per 128x128 block."""
    out, inp = weight.shape
    if out % FP8_BLOCK or inp % FP8_BLOCK:
        raise ValueError(f"FP8 weight {tuple(weight.shape)} is not {FP8_BLOCK}-aligned")
    blocks = weight.float().view(
        out // FP8_BLOCK, FP8_BLOCK, inp // FP8_BLOCK, FP8_BLOCK
    )
    return (blocks * scale.float()[:, None, :, None]).view(out, inp).to(torch.bfloat16)


def dequantize_fp4(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """DeepSeek's FP4 layout: e2m1 pairs ``[out, in // 2]`` with one e8m0 scale per 32 inputs."""
    codes = weight.view(torch.uint8)
    out, half = codes.shape
    table = _FP4_VALUES.to(codes.device)
    values = torch.empty(out, half * 2, dtype=torch.float32, device=codes.device)
    values[:, 0::2] = table[(codes & 0xF).long()]
    values[:, 1::2] = table[(codes >> 4).long()]
    groups = values.view(out, half * 2 // FP4_GROUP, FP4_GROUP)
    return (groups * scale.float()[:, :, None]).view(out, half * 2).to(torch.bfloat16)


def _dequantized(tensor: torch.Tensor, scale: Optional[torch.Tensor]) -> torch.Tensor:
    if scale is None:
        return tensor
    if tensor.dtype == torch.float8_e4m3fn:
        return dequantize_fp8(tensor, scale)
    if tensor.dtype in (torch.int8, torch.uint8):
        return dequantize_fp4(tensor, scale)
    raise ValueError(f"no dequantization for a {tensor.dtype} weight with a scale")


def _specforge_name(stage: int, rest: str, last_stage: int) -> str:
    if rest in _LAYER_NAMES:
        return f"layers.{stage}.{_LAYER_NAMES[rest]}"
    if stage == 0 and rest in _FIRST_STAGE_NAMES:
        return _FIRST_STAGE_NAMES[rest]
    if stage == last_stage and rest in _LAST_STAGE_NAMES:
        return _LAST_STAGE_NAMES[rest]
    raise KeyError(f"unexpected official tensor {OFFICIAL_PREFIX}{stage}.{rest}")


def import_official(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """The SpecForge state dict for official ``mtp.*`` tensors (scales consumed, experts fused)."""
    stages = sorted(
        {int(k.split(".")[1]) for k in tensors if k.startswith(OFFICIAL_PREFIX)}
    )
    if stages != list(range(len(stages))) or not stages:
        raise ValueError(f"expected contiguous {OFFICIAL_PREFIX} stages, got {stages}")
    last_stage = stages[-1]
    state: Dict[str, torch.Tensor] = {}
    experts: Dict[Tuple[int, str], Dict[int, torch.Tensor]] = {}
    for name, tensor in tensors.items():
        if not name.startswith(OFFICIAL_PREFIX) or name.endswith(".scale"):
            continue
        stage_text, rest = name[len(OFFICIAL_PREFIX) :].split(".", 1)
        stage = int(stage_text)
        scale = (
            tensors.get(name[: -len(".weight")] + ".scale")
            if name.endswith(".weight")
            else None
        )
        value = _dequantized(tensor, scale)
        expert = _EXPERT.fullmatch(rest)
        if expert:
            experts.setdefault((stage, expert.group(2)), {})[
                int(expert.group(1))
            ] = value
            continue
        state[_specforge_name(stage, rest, last_stage)] = value
    for stage in stages:
        w1, w2, w3 = (experts.get((stage, name), {}) for name in ("w1", "w2", "w3"))
        count = len(w1)
        if not count or any(sorted(w) != list(range(count)) for w in (w1, w2, w3)):
            raise ValueError(f"stage {stage} has an incomplete set of experts")
        state[f"layers.{stage}.mlp.experts.gate_up_proj"] = torch.stack(
            [torch.cat([w1[e], w3[e]]) for e in range(count)]
        )
        state[f"layers.{stage}.mlp.experts.down_proj"] = torch.stack(
            [w2[e] for e in range(count)]
        )
    return state


def export_official(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Official ``mtp.*`` tensors, unquantized, for a SpecForge state dict."""
    layers = sorted({int(m.group(1)) for k in state if (m := _LAYER.match(k))})
    last_stage = layers[-1]
    inverse_layer = {v: k for k, v in _LAYER_NAMES.items()}
    inverse_first = {v: k for k, v in _FIRST_STAGE_NAMES.items()}
    inverse_last = {v: k for k, v in _LAST_STAGE_NAMES.items()}
    official: Dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        match = _LAYER.match(name)
        if match:
            stage, rest = int(match.group(1)), match.group(2)
            if rest == "mlp.experts.gate_up_proj":
                for index, fused in enumerate(tensor):
                    w1, w3 = fused.chunk(2, dim=0)
                    official[
                        f"{OFFICIAL_PREFIX}{stage}.ffn.experts.{index}.w1.weight"
                    ] = w1
                    official[
                        f"{OFFICIAL_PREFIX}{stage}.ffn.experts.{index}.w3.weight"
                    ] = w3
            elif rest == "mlp.experts.down_proj":
                for index, w2 in enumerate(tensor):
                    official[
                        f"{OFFICIAL_PREFIX}{stage}.ffn.experts.{index}.w2.weight"
                    ] = w2
            elif rest in inverse_layer:
                official[f"{OFFICIAL_PREFIX}{stage}.{inverse_layer[rest]}"] = tensor
            else:
                raise KeyError(f"unexpected SpecForge tensor {name}")
        elif name in inverse_first:
            official[f"{OFFICIAL_PREFIX}0.{inverse_first[name]}"] = tensor
        elif name in inverse_last:
            official[f"{OFFICIAL_PREFIX}{last_stage}.{inverse_last[name]}"] = tensor
        else:
            raise KeyError(f"unexpected SpecForge tensor {name}")
    for name, tensor in official.items():
        rest = name.split(".", 2)[2]
        dtype = torch.float32 if rest.startswith(_FP32_OFFICIAL) else torch.bfloat16
        official[name] = tensor.detach().to(dtype).contiguous()
    return official


def official_draft_config(official_config: dict) -> dict:
    """The official config for unquantized ``mtp.*`` tensors: no quantization fields."""
    return {
        key: value
        for key, value in official_config.items()
        if key not in ("quantization_config", "expert_dtype")
    }


def official_tensors(
    source: str, cache_dir: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """The ``mtp.*`` tensors of a local or Hub checkpoint, fetching only their shards."""
    from specforge.modeling.target.checkpoint import (
        load_checkpoint_tensors,
        read_weight_map,
        resolve_checkpoint_dir,
    )

    if not os.path.exists(source):
        index_dir = resolve_checkpoint_dir(
            source, cache_dir=cache_dir, allow_patterns=["*.json"]
        )
        shards = sorted(
            {
                filename
                for key, filename in read_weight_map(index_dir).items()
                if key.startswith(OFFICIAL_PREFIX)
            }
        )
        source = resolve_checkpoint_dir(
            source, cache_dir=cache_dir, allow_patterns=["*.json", *shards]
        )
    return load_checkpoint_tensors(
        source, key_filter=lambda key: key.startswith(OFFICIAL_PREFIX)
    )


def check_against_draft(state: Dict[str, torch.Tensor], config: dict) -> None:
    """Every tensor the SpecForge draft of ``config`` expects, with its shape, and nothing else."""
    from transformers import Qwen3Config

    from specforge.modeling.draft.dspark import DSparkDraftModel

    with torch.device("meta"):
        model = DSparkDraftModel(Qwen3Config(**config))
    expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    actual = {key: tuple(value.shape) for key, value in state.items()}
    if expected != actual:
        missing = sorted(expected.keys() - actual.keys())
        unexpected = sorted(actual.keys() - expected.keys())
        mismatched = sorted(
            key
            for key in expected.keys() & actual.keys()
            if expected[key] != actual[key]
        )
        raise ValueError(
            "converted tensors do not match the draft config: "
            f"missing {missing[:5]}, unexpected {unexpected[:5]}, shapes {mismatched[:5]}"
        )


def write_checkpoint(
    state: Dict[str, torch.Tensor], output_dir: str, max_shard_size: str = "5GB"
) -> None:
    """Write safetensors shards (and an index when sharded) in the Hub layout."""
    from huggingface_hub import split_torch_state_dict_into_shards
    from safetensors.torch import save_file

    os.makedirs(output_dir, exist_ok=True)
    split = split_torch_state_dict_into_shards(state, max_shard_size=max_shard_size)
    for filename, keys in split.filename_to_tensors.items():
        shard = {key: state[key].contiguous() for key in keys}
        save_file(shard, os.path.join(output_dir, filename), metadata={"format": "pt"})
    if split.is_sharded:
        index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
        with open(
            os.path.join(output_dir, "model.safetensors.index.json"), "w"
        ) as handle:
            json.dump(index, handle, indent=2, sort_keys=True)


def _write_config(config: dict, output_dir: str) -> None:
    with open(os.path.join(output_dir, "config.json"), "w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def convert_official_to_specforge(
    source: str,
    output_dir: str,
    draft_config: str,
    *,
    cache_dir: Optional[str] = None,
    max_shard_size: str = "5GB",
) -> None:
    """Write a SpecForge draft directory holding the official DSpark draft's weights."""
    with open(draft_config) as handle:
        config = json.load(handle)
    state = import_official(official_tensors(source, cache_dir=cache_dir))
    check_against_draft(state, config)
    write_checkpoint(state, output_dir, max_shard_size)
    _write_config(config, output_dir)


def convert_specforge_to_official(
    draft_dir: str,
    output_dir: str,
    official_config: str,
    *,
    max_shard_size: str = "5GB",
) -> None:
    """Write the official ``mtp.*`` layout for SGLang's DeepseekV4ForCausalLMDSpark."""
    from specforge.modeling.target.checkpoint import load_checkpoint_tensors

    with open(official_config) as handle:
        config = official_draft_config(json.load(handle))
    write_checkpoint(
        export_official(load_checkpoint_tensors(draft_dir)), output_dir, max_shard_size
    )
    _write_config(config, output_dir)
