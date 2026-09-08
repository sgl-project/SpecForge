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
_DSPARK_DENSE_ARCHITECTURE = "Qwen3DSparkModel"
# SGLang draft model for Qwen3-style layers with a Qwen3-MoE FFN (softmax
# top-k routing, sigmoid-gated shared expert, folded router centering bias):
# sglang/srt/models/dspark_moe.py.
_DSPARK_MOE_ARCHITECTURE = "Qwen3MoeDSparkModel"
_DSPARK_MOE_REQUIRED_FIELDS = (
    "num_experts_per_tok",
    "moe_intermediate_size",
)
_DFLASH2_ARCHITECTURE = "DFlash2DraftModel"
_DFLASH2_FIELDS = (
    "conv_group_size",
    "conv_kernel_size",
    "selector_rank",
    "selector_top_k",
)


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


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
    if config.get("n_routed_experts") or config.get("num_experts"):
        _normalize_dspark_moe(config)
    else:
        config["architectures"] = [_DSPARK_DENSE_ARCHITECTURE]


def _normalize_dspark_moe(config: Dict[str, Any]) -> None:
    """Name the MoE DSpark architecture and pin the routing recipe it serves.

    SpecForge's HF export writes the resolved ``qwen3_5_moe`` preset in the
    DeepSeek config vocabulary (``n_routed_experts``, ``scoring_func``,
    ``norm_topk_prob``, ...) plus the Qwen ``shared_expert_intermediate_size``;
    SGLang's ``Qwen3MoeDSparkModel`` reads those and the Qwen alias
    ``num_experts``, and supports exactly the Qwen3-MoE recipe: softmax scores,
    plain (non-grouped) top-k, at most one sigmoid-gated shared expert.
    """
    num_experts = config.get("n_routed_experts", config.get("num_experts"))
    if not _positive_integer(num_experts):
        raise ValueError(
            f"DSpark MoE export requires a positive integer n_routed_experts, got {num_experts!r}"
        )
    alias = config.get("num_experts")
    if alias is not None and alias != num_experts:
        raise ValueError(
            f"DSpark MoE config conflict: num_experts {alias!r} != n_routed_experts {num_experts!r}"
        )
    for key in _DSPARK_MOE_REQUIRED_FIELDS:
        value = config.get(key)
        if not _positive_integer(value):
            raise ValueError(
                f"DSpark MoE export requires a positive integer {key}, got {value!r}"
            )
    if config["num_experts_per_tok"] > num_experts:
        raise ValueError(
            f"DSpark MoE export has num_experts_per_tok={config['num_experts_per_tok']} "
            f"> n_routed_experts={num_experts}"
        )
    scoring_func = str(config.get("scoring_func", "softmax")).lower()
    if scoring_func != "softmax":
        raise ValueError(
            f"{_DSPARK_MOE_ARCHITECTURE} routes with softmax scores, got scoring_func={scoring_func!r}"
        )
    topk_method = str(config.get("topk_method", "greedy")).lower()
    if topk_method not in {"greedy", "none"}:
        raise ValueError(
            f"{_DSPARK_MOE_ARCHITECTURE} supports plain top-k routing, got topk_method={topk_method!r}"
        )
    if (config.get("n_group") or 1) > 1:
        raise ValueError(
            f"{_DSPARK_MOE_ARCHITECTURE} does not support group-limited routing (n_group={config.get('n_group')!r})"
        )
    shared_width = config.get("shared_expert_intermediate_size", 0) or 0
    n_shared = config.get("n_shared_experts")
    if n_shared is None:
        n_shared = 1 if shared_width > 0 else 0
    if n_shared not in (0, 1):
        raise ValueError(
            f"{_DSPARK_MOE_ARCHITECTURE} supports at most one shared expert, got n_shared_experts={n_shared!r}"
        )
    if n_shared == 1 and not _positive_integer(shared_width):
        raise ValueError(
            "DSpark MoE export with a shared expert requires a positive integer "
            f"shared_expert_intermediate_size, got {shared_width!r}"
        )
    config["n_routed_experts"] = num_experts
    config["num_experts"] = num_experts
    config["n_shared_experts"] = n_shared
    config["scoring_func"] = scoring_func
    config.setdefault("norm_topk_prob", True)
    config.setdefault("routed_scaling_factor", 1.0)
    # The router bias is the folded centering term; the loader requires it
    # whenever training centred the router (see dspark_moe.py).
    method_config = config.get("dflash_config") or {}
    config["moe_router_bias"] = (
        str(method_config.get("moe_router_center", "none")).lower() != "none"
    )
    config["architectures"] = [_DSPARK_MOE_ARCHITECTURE]


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
    elif _DFLASH2_ARCHITECTURE in (config.get("architectures") or []):
        _normalize_dflash2(config, method_config)
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
