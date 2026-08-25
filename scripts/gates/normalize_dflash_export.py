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

    block_size = config.get("block_size")
    if block_size != expected_block_size:
        raise ValueError(
            f"exported block_size={block_size!r}, expected {expected_block_size}"
        )
    method_config = config.get("dflash_config") or {}
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
    else:
        config["architectures"] = ["DFlashDraftModel"]
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
