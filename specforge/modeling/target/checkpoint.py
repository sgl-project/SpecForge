# coding=utf-8
"""Model-agnostic selective loading from local or Hugging Face checkpoints.

These helpers know nothing about any model family or key naming convention;
callers provide the keys or the predicate.  Both sharded checkpoints
(``*.safetensors.index.json``) and single-file checkpoints are supported.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Callable, Dict, Iterable, List, Optional

import torch
from safetensors import safe_open


def resolve_checkpoint_dir(
    path_or_repo: str,
    cache_dir: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
) -> str:
    """Return a local checkpoint directory, downloading from the Hub if needed."""

    if os.path.exists(path_or_repo):
        return path_or_repo
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=path_or_repo,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns or ["*.json", "*.safetensors", "*.bin"],
    )


def read_weight_map(checkpoint_dir: str) -> Dict[str, str]:
    """Return the ``weight_map`` of a sharded checkpoint, or {} if unsharded."""

    index_files = glob.glob(os.path.join(checkpoint_dir, "*.index.json"))
    if not index_files:
        return {}
    with open(index_files[0], "r") as f:
        index = json.load(f)
    return index.get("weight_map", {})


def list_checkpoint_keys(checkpoint_dir: str) -> List[str]:
    """List all tensor keys without loading tensor payloads."""

    weight_map = read_weight_map(checkpoint_dir)
    if weight_map:
        return sorted(weight_map.keys())
    for pattern in ("*.safetensors", "*.bin"):
        files = sorted(glob.glob(os.path.join(checkpoint_dir, pattern)))
        if files:
            target = files[0]
            if target.endswith(".safetensors"):
                with safe_open(target, framework="pt") as f:
                    return sorted(f.keys())
            state = torch.load(target, map_location="cpu", weights_only=True)
            return sorted(state.keys())
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")


def load_selected_tensors(
    checkpoint_dir: str,
    predicate: Callable[[str], bool],
) -> Dict[str, torch.Tensor]:
    """Load only the tensors whose key matches ``predicate``.

    Sharded checkpoints open just the shards that hold selected keys.
    """

    weight_map = read_weight_map(checkpoint_dir)
    selected: Dict[str, torch.Tensor] = {}
    if weight_map:
        shards = sorted({weight_map[k] for k in weight_map if predicate(k)})
        for shard in shards:
            shard_path = os.path.join(checkpoint_dir, shard)
            if not os.path.exists(shard_path):
                continue
            with safe_open(shard_path, framework="pt") as f:
                for key in f.keys():
                    if predicate(key):
                        selected[key] = f.get_tensor(key)
        return selected

    for pattern in ("*.safetensors", "*.bin"):
        files = sorted(glob.glob(os.path.join(checkpoint_dir, pattern)))
        if files:
            target = files[0]
            if target.endswith(".safetensors"):
                with safe_open(target, framework="pt") as f:
                    for key in f.keys():
                        if predicate(key):
                            selected[key] = f.get_tensor(key)
            else:
                state = torch.load(target, map_location="cpu", weights_only=True)
                for key, value in state.items():
                    if predicate(key):
                        selected[key] = value
            return selected
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")


def load_tensors_by_keys(
    checkpoint_dir: str, keys: Iterable[str]
) -> Dict[str, torch.Tensor]:
    """Load exactly ``keys`` (missing keys are simply absent from the result)."""

    wanted = set(keys)
    return load_selected_tensors(checkpoint_dir, lambda key: key in wanted)


def merge_state_into_checkpoint(
    base_checkpoint_dir: str,
    state: Dict[str, torch.Tensor],
    output_dir: str,
    *,
    shard_name: str,
    drop_prefixes: Iterable[str] = (),
) -> None:
    """Merge a state dict into a copy of a base checkpoint (model-agnostic).

    Copies non-weight files, drops base weight entries under ``drop_prefixes``,
    and merges ``state``.  Sharded bases get ``state`` written to a new
    ``shard_name`` shard with the index ``weight_map`` updated in place (the
    large base shards are never rewritten); single-file bases are rewritten
    whole under their original file name.
    """

    import shutil

    from safetensors.torch import save_file

    os.makedirs(output_dir, exist_ok=True)
    prefixes = tuple(drop_prefixes)

    # Copy non-weight files so the output directory is self-contained.
    for fname in os.listdir(base_checkpoint_dir):
        src = os.path.join(base_checkpoint_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, fname))

    index_files = glob.glob(os.path.join(base_checkpoint_dir, "*.index.json"))
    if index_files:
        with open(index_files[0], "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})

        old_keys = [k for k in weight_map if k.startswith(prefixes)]
        for key in old_keys:
            del weight_map[key]
        if old_keys:
            print(
                f"Replaced {len(old_keys)} weight entries under {prefixes} "
                "from base model."
            )

        # Write the incoming tensors to a dedicated shard; base shards untouched.
        save_file(state, os.path.join(output_dir, shard_name))
        for key in state.keys():
            weight_map[key] = shard_name

        index["weight_map"] = weight_map
        with open(os.path.join(output_dir, os.path.basename(index_files[0])), "w") as f:
            json.dump(index, f, indent=2)
        return

    # Single-file base: load, drop, merge, rewrite under the original name.
    base_safetensors = glob.glob(os.path.join(base_checkpoint_dir, "*.safetensors"))
    base_bins = glob.glob(os.path.join(base_checkpoint_dir, "*.bin"))
    if not base_safetensors and not base_bins:
        raise FileNotFoundError(f"No checkpoint found in {base_checkpoint_dir}")
    base_state = (
        load_selected_tensors(base_checkpoint_dir, lambda _key: True)
        if base_safetensors
        else torch.load(base_bins[0], map_location="cpu", weights_only=True)
    )
    out_name = os.path.basename(
        base_safetensors[0] if base_safetensors else base_bins[0]
    )

    old_keys = [k for k in base_state if k.startswith(prefixes)]
    for key in old_keys:
        del base_state[key]
    if old_keys:
        print(
            f"Replaced {len(old_keys)} weight entries under {prefixes} "
            "from base model."
        )

    merged = {**base_state, **state}
    if out_name.endswith(".safetensors"):
        save_file(merged, os.path.join(output_dir, out_name))
    else:
        torch.save(merged, os.path.join(output_dir, out_name))


__all__ = [
    "list_checkpoint_keys",
    "load_selected_tensors",
    "load_tensors_by_keys",
    "merge_state_into_checkpoint",
    "read_weight_map",
    "resolve_checkpoint_dir",
]
