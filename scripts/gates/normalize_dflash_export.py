"""Normalize a SpecForge DFlash-family HF export for SGLang loading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

_DSPARK_MARKOV_HEAD_TYPES = frozenset({"vanilla", "gated", "rnn"})
_DSPARK_TOP_LEVEL_FIELDS = (
    "markov_rank",
    "markov_head_type",
    "enable_confidence_head",
    "confidence_head_with_markov",
)
_DFLASH2_ARCHITECTURE = "DFlash2DraftModel"
_DFLASH2_FIELDS = (
    "conv_group_size",
    "conv_kernel_size",
    "selector_rank",
    "selector_top_k",
)
# Per moe_preset: the specforge.serving.sglang_models class that serves each
# draft kind, and the config fields that class builds its MoE FFN from.
_MOE_SERVED = {
    "qwen3_5_moe": {
        "dspark": "Qwen3MoeDSparkModel",
        "dflash": "DFlashMoEDraftModel",
        "dflash2": "DFlash2MoEDraftModel",  # needs SGLang >= 0.5.19 (DFlash2 serving)
        "fields": ("num_experts", "num_experts_per_tok", "moe_intermediate_size"),
    },
}


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _normalize_moe(config: Dict[str, Any], kind: str) -> None:
    """Name the class that serves ``kind`` MoE drafts; refuse exports nothing serves."""
    preset = config.get("moe_preset")
    if not preset:
        return
    served = _MOE_SERVED.get(preset, {})
    if kind not in served:
        raise ValueError(
            f"no SGLang model serves a {kind} draft with moe_preset={preset!r}"
        )
    for key in served["fields"]:
        if not _positive_integer(config.get(key)):
            raise ValueError(
                f"moe_preset={preset!r} export requires a positive integer {key}, "
                f"got {config.get(key)!r}"
            )
    # Loaded through SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang_models.
    config["architectures"] = [served[kind]]


def _normalize_dflash2(config: Dict[str, Any], method_config: Dict[str, Any]) -> None:
    for key in _DFLASH2_FIELDS:
        value = method_config.get(key)
        if not _positive_integer(value):
            raise ValueError(
                f"DFlash2 export requires a positive integer dflash_config.{key}, "
                f"got {value!r}"
            )
    config["architectures"] = [_DFLASH2_ARCHITECTURE]


def _normalize_dspark(config: Dict[str, Any], method_config: Dict[str, Any]) -> None:
    if config.get("model_type") != "qwen3":
        raise ValueError(
            "SGLang's standalone DSpark export expects model_type='qwen3', "
            f"got {config.get('model_type')!r}"
        )

    markov_rank = method_config.get("markov_rank", config.get("markov_rank", 0))
    if (
        not isinstance(markov_rank, int)
        or isinstance(markov_rank, bool)
        or markov_rank <= 0
    ):
        raise ValueError(
            "DSpark export requires a positive integer markov_rank, "
            f"got {markov_rank!r}"
        )

    markov_head_type = method_config.get(
        "markov_head_type", config.get("markov_head_type")
    )
    if (
        not isinstance(markov_head_type, str)
        or markov_head_type.lower() not in _DSPARK_MARKOV_HEAD_TYPES
    ):
        raise ValueError(
            "DSpark export requires markov_head_type to be one of "
            f"{sorted(_DSPARK_MARKOV_HEAD_TYPES)}, got {markov_head_type!r}"
        )

    for key in _DSPARK_TOP_LEVEL_FIELDS:
        nested_value = method_config.get(key)
        if nested_value is None:
            continue
        top_level_value = config.get(key)
        if top_level_value is not None and top_level_value != nested_value:
            raise ValueError(
                f"DSpark config conflict for {key}: top-level "
                f"{top_level_value!r} != dflash_config {nested_value!r}"
            )
        config[key] = nested_value

    # The two required fields may already be top-level rather than nested.
    config["markov_rank"] = markov_rank
    config["markov_head_type"] = markov_head_type.lower()
    config["architectures"] = ["Qwen3DSparkModel"]


def normalize_export(config_path: str, expected_block_size: int) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    method_config = config.get("dflash_config") or {}
    top_level_block_size = config.get("block_size")
    nested_block_size = method_config.get("block_size")
    if (
        top_level_block_size is not None
        and nested_block_size is not None
        and top_level_block_size != nested_block_size
    ):
        raise ValueError(
            "exported block_size conflict: top-level "
            f"{top_level_block_size!r} != dflash_config {nested_block_size!r}"
        )
    block_size = (
        top_level_block_size if top_level_block_size is not None else nested_block_size
    )
    if block_size != expected_block_size:
        raise ValueError(
            f"exported block_size={block_size!r}, expected {expected_block_size}"
        )
    projector_type = method_config.get("projector_type", "dflash")
    if projector_type not in {"dflash", "domino", "dspark"}:
        raise ValueError(
            "export is not DFlash-family: "
            f"dflash_config.projector_type={projector_type!r}"
        )
    attention_mode = method_config.get("attention_mode", "gqa")
    if not isinstance(attention_mode, str) or attention_mode.lower() not in {
        "gqa",
        "mha",
    }:
        raise ValueError(
            "SGLang DFlash-family serving supports only GQA/MHA exports, "
            f"got dflash_config.attention_mode={attention_mode!r}"
        )

    if projector_type == "dspark":
        _normalize_dspark(config, method_config)
        _normalize_moe(config, "dspark")
    elif _DFLASH2_ARCHITECTURE in (config.get("architectures") or []):
        _normalize_dflash2(config, method_config)
        _normalize_moe(config, "dflash2")
    else:
        config["architectures"] = ["DFlashDraftModel"]
        _normalize_moe(config, "dflash")
    config.pop("auto_map", None)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--block-size", type=int, required=True)
    args = parser.parse_args()
    normalize_export(args.config, args.block_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
