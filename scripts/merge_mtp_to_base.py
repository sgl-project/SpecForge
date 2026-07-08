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
from typing import Dict, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig


def load_safetensors_file(path: str) -> Dict[str, torch.Tensor]:
    """Load all tensors from a safetensors file."""
    state = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)
    return state


def convert_mtp_keys(state_dict: Dict[str, torch.Tensor], fmt: str) -> Dict[str, torch.Tensor]:
    """Convert MTP weight keys to the requested output format.

    SpecForge training saves keys as:
        mtp.model.layers.0.self_attn.q_proj.weight
        mtp.pre_fc_norm_embedding.weight
        ...

    Supported output formats:
      - sglang: keep the training layout (mtp.model.layers.0.*). SGLang's
        Qwen3_5ForCausalLMMTP remaps the leading 'mtp.' -> 'model.' internally.
      - hf: convert to the HuggingFace / vLLM native layout:
          mtp.model.layers.0.* -> mtp.layers.0.*
    """
    if fmt == "sglang":
        return dict(state_dict)

    if fmt == "hf":
        converted = {}
        for k, v in state_dict.items():
            if k.startswith("mtp.model.layers."):
                new_k = k.replace("mtp.model.layers.", "mtp.layers.", 1)
            else:
                new_k = k
            converted[new_k] = v
        return converted

    raise ValueError(f"Unknown key format: {fmt}")


def _is_mtp_key(key: str) -> bool:
    """Return True for keys that belong to the MTP module."""
    return key.startswith("mtp.")


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
            print(f"Replaced {len(old_mtp_keys)} native MTP weight entries from base model.")

        # Write MTP weights into a dedicated shard so we do not need to rewrite
        # the (large) base model shards.
        mtp_shard_name = "mtp-merged.safetensors"
        save_file(mtp_state, os.path.join(output_path, mtp_shard_name))

        # Update the weight map with the new MTP keys.
        for key in mtp_state.keys():
            weight_map[key] = mtp_shard_name

        # Save the updated index. Re-use the original index file name.
        index["weight_map"] = weight_map
        with open(os.path.join(output_path, os.path.basename(index_files[0])), "w") as f:
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
            print(f"Replaced {len(old_mtp_keys)} native MTP weight entries from base model.")

        merged = {**base_state, **mtp_state}

        if out_name.endswith(".safetensors"):
            save_file(merged, os.path.join(output_path, out_name))
        else:
            torch.save(merged, os.path.join(output_path, out_name))

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
            "Target key layout. 'sglang' keeps mtp.model.layers.* (matches "
            "SGLang's internal remap). 'hf' converts to mtp.layers.* (matches "
            "the native HuggingFace / vLLM layout)."
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
