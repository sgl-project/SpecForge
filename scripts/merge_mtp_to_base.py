#!/usr/bin/env python3
# coding=utf-8
"""Merge a trained MTP draft checkpoint back into the base target model.

This produces a single model directory that contains both the base Qwen3.5
weights and the trained MTP weights, so it can be served directly without a
separate draft-model path.

Example:
    python scripts/merge_mtp_to_base.py \
        --base-model-path PATH/TO/Qwen3.5-4B \
        --mtp-checkpoint-path PATH/TO/outputs/qwen3.5-4b-mtp/epoch_0_step_X \
        --output-path PATH/TO/Qwen3.5-4B-MTP \
        --key-format sglang
"""

import argparse
import glob
import json
import os
import shutil
from typing import Dict

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def load_safetensors_file(path: str) -> Dict[str, torch.Tensor]:
    """Load all tensors from a safetensors file."""
    state = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)
    return state


def convert_mtp_keys(
    state_dict: Dict[str, torch.Tensor], fmt: str
) -> Dict[str, torch.Tensor]:
    """Convert MTP weight keys to the requested output format.

    SpecForge training saves keys in the flat native layout:
        mtp.layers.0.self_attn.q_proj.weight
        mtp.pre_fc_norm_embedding.weight
        mtp.norm.weight
        ...

    This flat layout is what both SGLang's ``Qwen3_5ForCausalLMMTP.load_weights``
    (``mtp.`` -> ``model.`` remap, flat ``Qwen3_5ForCausalLM``) and the native
    HuggingFace / vLLM MTP modules expect, so both formats are identical.
    The ``fmt`` argument is kept for backward compatibility; ``sglang`` and
    ``hf`` both return the flat layout unchanged. A legacy nested layout
    (``mtp.model.layers.0.*``) is also normalized to flat if encountered.
    """
    converted = {}
    for k, v in state_dict.items():
        # Normalize legacy nested keys (mtp.model.layers.* -> mtp.layers.*).
        if k.startswith("mtp.model.layers."):
            new_k = k.replace("mtp.model.layers.", "mtp.layers.", 1)
        elif k == "mtp.model.norm.weight":
            new_k = "mtp.norm.weight"
        # Promote bare embed_tokens / lm_head saved by the training script to the
        # ``mtp.*`` namespace expected by vLLM/SGLang.
        elif k == "embed_tokens.weight":
            new_k = "mtp.embed_tokens.weight"
        elif k == "lm_head.weight":
            new_k = "mtp.lm_head.weight"
        else:
            new_k = k
        converted[new_k] = v
    return converted


def _is_mtp_key(key: str) -> bool:
    """Return True for keys that belong to the MTP module."""
    return key.startswith("mtp.")


def _find_base_key(state_dict: Dict[str, torch.Tensor], *candidates: str) -> str | None:
    """Return the first candidate key that exists in ``state_dict``."""
    for key in candidates:
        if key in state_dict:
            return key
    return None


def _copy_shared_embeddings(
    base_state: Dict[str, torch.Tensor],
    mtp_state: Dict[str, torch.Tensor],
    tie_word_embeddings: bool,
) -> Dict[str, torch.Tensor]:
    """Copy base embed_tokens/lm_head into the MTP state if they are missing.

    During training the draft model typically shares ``embed_tokens`` and
    ``lm_head`` with the target model, so the saved MTP checkpoint does not
    contain those tensors.  vLLM/SGLang, however, instantiate their own
    ``mtp.embed_tokens`` (and a separate ``lm_head`` when weights are not tied),
    and expect them in the checkpoint.  Copying them from the base model keeps
    the merged checkpoint self-contained and avoids random-initialization of the
    MTP input/output embeddings at serving time.
    """
    if "mtp.embed_tokens.weight" not in mtp_state:
        embed_key = _find_base_key(
            base_state,
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
        )
        if embed_key:
            mtp_state["mtp.embed_tokens.weight"] = base_state[embed_key]
            print(f"  copied {embed_key} -> mtp.embed_tokens.weight")
        else:
            print(
                "  warning: base embed_tokens.weight not found; "
                "mtp.embed_tokens will be randomly initialized"
            )

    if not tie_word_embeddings and "mtp.lm_head.weight" not in mtp_state:
        lm_head_key = _find_base_key(
            base_state,
            "model.language_model.lm_head.weight",
            "model.lm_head.weight",
            "lm_head.weight",
        )
        if lm_head_key:
            mtp_state["mtp.lm_head.weight"] = base_state[lm_head_key]
            print(f"  copied {lm_head_key} -> mtp.lm_head.weight")
        else:
            print(
                "  warning: base lm_head.weight not found; "
                "mtp.lm_head will be randomly initialized"
            )

    return mtp_state


def _patch_text_config(base_config: dict, draft_config: dict) -> dict:
    """Ensure base text_config contains MTP-critical dims from the draft config.

    Some Qwen3.5 base checkpoints omit ``head_dim`` in ``text_config``; vLLM's
    ``Qwen3_5TextConfig`` then falls back to its default (``head_dim=256``),
    which mismatches the trained MTP weights (e.g. q_norm/k_norm shape 128).
    We sync only the structural dims that must agree between base and draft.
    """
    keys_to_sync = [
        "head_dim",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
    ]

    target = base_config
    if "text_config" in base_config:
        target = base_config["text_config"]

    source = draft_config
    if "text_config" in draft_config:
        source = draft_config["text_config"]

    for key in keys_to_sync:
        if key not in source:
            continue
        old = target.get(key)
        new = source[key]
        if old != new:
            target[key] = new
            print(f"  overriding text_config.{key}: {old} -> {new}")

    return base_config


def merge_checkpoints(
    base_model_path: str,
    mtp_checkpoint_path: str,
    output_path: str,
    key_format: str = "sglang",
):
    os.makedirs(output_path, exist_ok=True)

    # Copy non-weight files from the base model so the output directory is a
    # fully self-contained HF checkpoint.
    for fname in os.listdir(base_model_path):
        src = os.path.join(base_model_path, fname)
        dst = os.path.join(output_path, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    # Load MTP weights. The training script saves them as a draft_model
    # checkpoint: model.safetensors + config.json + mtp.py.
    mtp_safetensors = glob.glob(os.path.join(mtp_checkpoint_path, "*.safetensors"))
    mtp_bins = glob.glob(os.path.join(mtp_checkpoint_path, "*.bin"))
    if mtp_safetensors:
        mtp_state = load_safetensors_file(mtp_safetensors[0])
    elif mtp_bins:
        mtp_state = torch.load(mtp_bins[0], map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(
            f"No safetensors/bin weights found in {mtp_checkpoint_path}"
        )

    mtp_state = convert_mtp_keys(mtp_state, key_format)

    # Determine whether word embeddings are tied so we know whether a separate
    # lm_head must be materialized for the MTP module.
    tie_word_embeddings = True
    base_config_path = os.path.join(base_model_path, "config.json")
    if os.path.exists(base_config_path):
        with open(base_config_path, "r") as f:
            base_cfg = json.load(f)
        # VLM checkpoints nest text config under "text_config".
        text_cfg = base_cfg.get("text_config", base_cfg)
        tie_word_embeddings = text_cfg.get("tie_word_embeddings", True)

    # Load base model index if present.
    index_files = glob.glob(os.path.join(base_model_path, "*.index.json"))
    if index_files:
        with open(index_files[0], "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})

        # If the base checkpoint already contains native MTP weights (e.g. an
        # official Qwen3.5 checkpoint), drop the old MTP entries so the trained
        # MTP weights replace them rather than duplicating them.
        old_mtp_keys = [k for k in weight_map if _is_mtp_key(k)]
        for key in old_mtp_keys:
            del weight_map[key]
        if old_mtp_keys:
            print(
                f"Replaced {len(old_mtp_keys)} native MTP weight entries from base model."
            )

        # If the trained checkpoint did not save shared embeddings, copy them
        # from the base model shards so vLLM/SGLang can initialise the MTP
        # embed_tokens/lm_head from the merged checkpoint.
        if "mtp.embed_tokens.weight" not in mtp_state or (
            not tie_word_embeddings and "mtp.lm_head.weight" not in mtp_state
        ):
            base_state: Dict[str, torch.Tensor] = {}
            for candidate in [
                "model.language_model.embed_tokens.weight",
                "model.embed_tokens.weight",
                "embed_tokens.weight",
                "model.language_model.lm_head.weight",
                "model.lm_head.weight",
                "lm_head.weight",
            ]:
                if candidate not in weight_map:
                    continue
                shard_path = os.path.join(base_model_path, weight_map[candidate])
                if not os.path.exists(shard_path):
                    continue
                with safe_open(shard_path, framework="pt") as f:
                    if candidate in f.keys():
                        base_state[candidate] = f.get_tensor(candidate)
            mtp_state = _copy_shared_embeddings(
                base_state, mtp_state, tie_word_embeddings
            )

        # Write MTP weights into a dedicated shard so we do not need to rewrite
        # the (large) base model shards.
        mtp_shard_name = "mtp-merged.safetensors"
        save_file(mtp_state, os.path.join(output_path, mtp_shard_name))

        # Update the weight map with the new MTP keys.
        for key in mtp_state.keys():
            weight_map[key] = mtp_shard_name

        # Save the updated index. Re-use the original index file name.
        index["weight_map"] = weight_map
        with open(
            os.path.join(output_path, os.path.basename(index_files[0])), "w"
        ) as f:
            json.dump(index, f, indent=2)
    else:
        # Single-file checkpoint: load base weights, merge, and rewrite.
        base_safetensors = glob.glob(os.path.join(base_model_path, "*.safetensors"))
        base_bins = glob.glob(os.path.join(base_model_path, "*.bin"))

        if base_safetensors:
            base_state = load_safetensors_file(base_safetensors[0])
            out_name = os.path.basename(base_safetensors[0])
        elif base_bins:
            base_state = torch.load(base_bins[0], map_location="cpu", weights_only=True)
            out_name = os.path.basename(base_bins[0])
        else:
            raise FileNotFoundError(f"No checkpoint found in {base_model_path}")

        # Remove any native MTP weights from the base state before overwriting
        # with the trained MTP weights.
        old_mtp_keys = [k for k in base_state if _is_mtp_key(k)]
        for key in old_mtp_keys:
            del base_state[key]
        if old_mtp_keys:
            print(
                f"Replaced {len(old_mtp_keys)} native MTP weight entries from base model."
            )

        mtp_state = _copy_shared_embeddings(base_state, mtp_state, tie_word_embeddings)

        merged = {**base_state, **mtp_state}

        if out_name.endswith(".safetensors"):
            save_file(merged, os.path.join(output_path, out_name))
        else:
            torch.save(merged, os.path.join(output_path, out_name))

    # Ensure the merged config exposes the MTP structural dims.  vLLM/SGLang
    # use these values to build the MTP module; if the base config omits
    # ``head_dim`` (common for some Qwen3.5 checkpoints), the loader will use
    # its default and fail with a shape mismatch.
    draft_config_path = os.path.join(mtp_checkpoint_path, "config.json")
    output_config_path = os.path.join(output_path, "config.json")
    if os.path.exists(draft_config_path) and os.path.exists(output_config_path):
        with open(draft_config_path, "r") as f:
            draft_config = json.load(f)
        with open(output_config_path, "r") as f:
            base_config = json.load(f)
        patched_config = _patch_text_config(base_config, draft_config)
        with open(output_config_path, "w") as f:
            json.dump(patched_config, f, indent=2)

    # Copy over the MTP modeling file if present; some loaders need it for
    # trust_remote_code / auto_map resolution.
    mtp_py_src = os.path.join(mtp_checkpoint_path, "mtp.py")
    if os.path.exists(mtp_py_src):
        shutil.copy2(mtp_py_src, os.path.join(output_path, "mtp.py"))

    # If the base config has auto_map that points to the base modeling file, we
    # keep it. For SGLang/HF-native MTP inference the config.json from the base
    # model is usually sufficient.
    print(f"Merged checkpoint saved to {output_path}")
    print(f"  key format: {key_format}")
    print(f"  MTP tensors merged: {len(mtp_state)}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge trained MTP weights back into the base Qwen3.5 model."
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        required=True,
        help="Path to the original Qwen3.5 base model checkpoint.",
    )
    parser.add_argument(
        "--mtp-checkpoint-path",
        type=str,
        required=True,
        help="Path to a training output directory containing MTP weights.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Directory to write the merged checkpoint.",
    )
    parser.add_argument(
        "--key-format",
        type=str,
        default="sglang",
        choices=["sglang", "hf"],
        help=(
            "MTP key layout. Both 'sglang' and 'hf' produce the flat native "
            "layout (mtp.layers.0.* / mtp.norm.weight) that SGLang's flat "
            "Qwen3_5ForCausalLMMTP and HF/vLLM MTP modules expect; the argument "
            "is kept for backward compatibility."
        ),
    )
    args = parser.parse_args()

    merge_checkpoints(
        args.base_model_path,
        args.mtp_checkpoint_path,
        args.output_path,
        args.key_format,
    )


if __name__ == "__main__":
    main()
