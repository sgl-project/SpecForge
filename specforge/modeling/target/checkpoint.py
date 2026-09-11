# coding=utf-8
"""Model-agnostic selective loading from local or Hugging Face checkpoints.

These helpers know nothing about any model family or key naming convention.
The public tensor loader accepts exact keys, a key filter, or no selector for
an intentional full load. Both sharded checkpoints
(``*.safetensors.index.json``) and single-file checkpoints are supported.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from safetensors import safe_open


_CANONICAL_LAYOUTS = (
    ("single", "model.safetensors"),
    ("index", "model.safetensors.index.json"),
    ("single", "pytorch_model.bin"),
    ("index", "pytorch_model.bin.index.json"),
)


def _weight_artifact_paths(checkpoint_dir: str) -> set[str]:
    """Return top-level files that can participate in weight resolution."""

    paths: set[str] = set()
    for pattern in ("*.index.json", "*.safetensors", "*.bin"):
        paths.update(glob.glob(os.path.join(checkpoint_dir, pattern)))
    return paths


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


def _resolve_weight_layout(
    checkpoint_dir: str, *, allow_missing: bool = False
) -> Tuple[str, str]:
    """Return (kind, path) for one deterministic checkpoint layout.

    Hugging Face prefers safetensors when both safetensors and PyTorch weights
    are present, so use the same precedence for the canonical file names. A
    non-canonical layout is accepted only when it is unambiguous.
    """

    for kind, filename in _CANONICAL_LAYOUTS:
        path = os.path.join(checkpoint_dir, filename)
        if os.path.isfile(path):
            return kind, path

    index_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.index.json")))
    if len(index_files) == 1:
        return "index", index_files[0]
    if len(index_files) > 1:
        raise FileNotFoundError(
            f"Multiple checkpoint index files found in {checkpoint_dir}: "
            f"{[os.path.basename(path) for path in index_files]}"
        )

    weight_files = sorted(
        glob.glob(os.path.join(checkpoint_dir, "*.safetensors"))
        + glob.glob(os.path.join(checkpoint_dir, "*.bin"))
    )
    if len(weight_files) == 1:
        return "single", weight_files[0]
    if len(weight_files) > 1:
        raise FileNotFoundError(
            f"Multiple unindexed checkpoint files found in {checkpoint_dir}: "
            f"{[os.path.basename(path) for path in weight_files]}"
        )
    if allow_missing:
        return "missing", ""
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")


def _read_weight_map_file(index_path: str) -> Dict[str, str]:
    with open(index_path, encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Checkpoint index has no valid weight_map: {index_path}")
    if not all(
        isinstance(key, str) and isinstance(filename, str)
        for key, filename in weight_map.items()
    ):
        raise ValueError(f"Checkpoint index has an invalid weight_map: {index_path}")
    return weight_map


def _load_tensor_file(
    path: str, predicate: Callable[[str], bool]
) -> Dict[str, torch.Tensor]:
    """Load matching tensors from one safetensors or PyTorch weight file."""

    if path.endswith(".safetensors"):
        with safe_open(path, framework="pt", device="cpu") as handle:
            return {
                key: handle.get_tensor(key) for key in handle.keys() if predicate(key)
            }

    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint file does not contain a state dict: {path}")
    return {key: value for key, value in state.items() if predicate(key)}


def read_weight_map(checkpoint_dir: str) -> Dict[str, str]:
    """Return the weight map of a sharded checkpoint, or {} if unsharded."""

    try:
        kind, path = _resolve_weight_layout(checkpoint_dir, allow_missing=True)
    except FileNotFoundError:
        if not glob.glob(os.path.join(checkpoint_dir, "*.index.json")):
            return {}
        raise
    if kind != "index":
        return {}
    return _read_weight_map_file(path)


def list_checkpoint_keys(checkpoint_dir: str) -> List[str]:
    """List all tensor keys without loading tensor payloads."""

    kind, path = _resolve_weight_layout(checkpoint_dir)
    if kind == "index":
        return sorted(_read_weight_map_file(path))
    if path.endswith(".safetensors"):
        with safe_open(path, framework="pt", device="cpu") as handle:
            return sorted(handle.keys())
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint file does not contain a state dict: {path}")
    return sorted(state)


def load_checkpoint_tensors(
    path_or_repo: str,
    *,
    keys: Optional[Iterable[str]] = None,
    key_filter: Optional[Callable[[str], bool]] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Load tensors from a local or Hugging Face checkpoint.

    ``keys`` requests an exact set and raises if any requested key is missing.
    ``key_filter`` loads every matching tensor and permits zero matches. With
    neither selector, the entire checkpoint is loaded; this can materialize a
    large model in CPU memory, especially for PyTorch ``.bin`` checkpoints.
    ``keys`` and ``key_filter`` are mutually exclusive.

    Tensors retain their stored dtype and are always returned on CPU. Sharded
    checkpoints open only the shards containing selected keys.
    """

    if keys is not None and key_filter is not None:
        raise ValueError("keys and key_filter are mutually exclusive")

    wanted: Optional[set[str]] = None
    if keys is not None:
        if isinstance(keys, (str, bytes)):
            raise TypeError("keys must be an iterable of tensor names, not a string")
        wanted = set(keys)
        if not all(isinstance(key, str) for key in wanted):
            raise TypeError("keys must contain only tensor-name strings")
        if not wanted:
            return {}
        predicate = wanted.__contains__
    elif key_filter is not None:
        if not callable(key_filter):
            raise TypeError("key_filter must be callable")
        predicate = key_filter
    else:
        predicate = lambda _key: True

    checkpoint_dir = resolve_checkpoint_dir(path_or_repo, cache_dir=cache_dir)
    kind, path = _resolve_weight_layout(checkpoint_dir)
    selected: Dict[str, torch.Tensor] = {}
    if kind == "index":
        weight_map = _read_weight_map_file(path)
        keys_by_shard: Dict[str, set[str]] = {}
        for key, shard in weight_map.items():
            if predicate(key):
                keys_by_shard.setdefault(shard, set()).add(key)

        for shard, shard_keys in sorted(keys_by_shard.items()):
            shard_path = os.path.join(checkpoint_dir, shard)
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(
                    f"Checkpoint index {path} references missing shard {shard_path}"
                )
            shard_tensors = _load_tensor_file(
                shard_path, lambda key: key in shard_keys
            )
            missing_from_shard = sorted(shard_keys - shard_tensors.keys())
            if missing_from_shard:
                raise KeyError(
                    f"Checkpoint shard {shard_path} is missing indexed tensors: "
                    f"{missing_from_shard}"
                )
            selected.update(shard_tensors)
    else:
        selected = _load_tensor_file(path, predicate)

    if wanted is not None:
        missing = sorted(wanted - selected.keys())
        if missing:
            raise KeyError(
                f"Checkpoint {checkpoint_dir} is missing requested tensors: {missing}"
            )
    return selected


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

    if os.path.realpath(base_checkpoint_dir) == os.path.realpath(output_dir):
        raise ValueError("base_checkpoint_dir and output_dir must be different")

    os.makedirs(output_dir, exist_ok=True)
    prefixes = tuple(drop_prefixes)
    layout_kind, base_weight_path = _resolve_weight_layout(base_checkpoint_dir)

    # A pre-existing model.safetensors, index, or shard may outrank the layout
    # written below and make the merged checkpoint load stale weights. Refuse
    # that ambiguous destination instead of deleting user files implicitly.
    existing_weight_files = sorted(_weight_artifact_paths(output_dir))
    if existing_weight_files:
        raise FileExistsError(
            f"Output directory already contains checkpoint weight files: "
            f"{existing_weight_files}"
        )

    # Copy non-weight files so the output directory is self-contained. Only
    # the selected model-weight representation is written below; copying an
    # alternate representation would leave a stale model in the output.
    weight_files = {
        os.path.basename(path)
        for path in _weight_artifact_paths(base_checkpoint_dir)
    }
    for fname in os.listdir(base_checkpoint_dir):
        src = os.path.join(base_checkpoint_dir, fname)
        if os.path.isfile(src) and fname not in weight_files:
            shutil.copy2(src, os.path.join(output_dir, fname))

    if layout_kind == "index":
        with open(base_weight_path, encoding="utf-8") as f:
            index = json.load(f)
        weight_map = _read_weight_map_file(base_weight_path)
        shard_formats = {
            "safetensors" if name.endswith(".safetensors") else "bin"
            for name in weight_map.values()
        }
        if len(shard_formats) != 1:
            raise ValueError(
                f"Checkpoint index mixes weight formats: {base_weight_path}"
            )

        old_keys = [k for k in weight_map if k.startswith(prefixes)]
        for key in old_keys:
            del weight_map[key]
        if old_keys:
            print(
                f"Replaced {len(old_keys)} weight entries under {prefixes} "
                "from base model."
            )

        for base_shard in sorted(set(weight_map.values())):
            source_path = os.path.join(base_checkpoint_dir, base_shard)
            if not os.path.isfile(source_path):
                raise FileNotFoundError(
                    f"Checkpoint index {base_weight_path} references missing "
                    f"shard {source_path}"
                )
            destination_path = os.path.join(output_dir, base_shard)
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            shutil.copy2(source_path, destination_path)

        # Write the incoming tensors in the index's existing format; base
        # shards remain untouched.
        shard_format = shard_formats.pop()
        shard_suffix = ".safetensors" if shard_format == "safetensors" else ".bin"
        output_shard_name = os.path.splitext(shard_name)[0] + shard_suffix
        output_shard_path = os.path.join(output_dir, output_shard_name)
        if shard_format == "safetensors":
            save_file(state, output_shard_path)
        else:
            torch.save(state, output_shard_path)
        for key in state.keys():
            weight_map[key] = output_shard_name

        index["weight_map"] = weight_map
        with open(
            os.path.join(output_dir, os.path.basename(base_weight_path)),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(index, f, indent=2)
        return

    # Single-file base: load, drop, merge, rewrite under the original name.
    base_state = load_checkpoint_tensors(base_checkpoint_dir)
    out_name = os.path.basename(base_weight_path)

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
    "load_checkpoint_tensors",
    "merge_state_into_checkpoint",
    "read_weight_map",
    "resolve_checkpoint_dir",
]
