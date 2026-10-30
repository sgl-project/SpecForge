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
from typing import Callable, Dict, Iterable, List, Optional

import torch
from safetensors import safe_open

_CANONICAL_LAYOUTS = (
    ("single", "model.safetensors"),
    ("index", "model.safetensors.index.json"),
    ("single", "pytorch_model.bin"),
    ("index", "pytorch_model.bin.index.json"),
)


def _resolve_weight_layout(
    checkpoint_dir: str, *, allow_missing: bool = False
) -> tuple[str, str]:
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


_HUB_CACHE_REPO_PREFIX = "models--"


def _inside(path: str, root: str) -> bool:
    """Return whether ``path`` is ``root`` itself or lives below it.

    ``os.path.commonpath`` compares the text it is given, case included, so two
    spellings of the same Windows directory (``C:\\Temp`` vs ``c:\\temp``) would
    compare as unrelated. ``os.path.normcase`` folds that away on the platforms
    that need it and is the identity on POSIX, which keeps a real escape a
    mismatch on both.
    """

    try:
        return os.path.normcase(os.path.commonpath([path, root])) == os.path.normcase(
            root
        )
    except ValueError:
        # Different drives on Windows: no common path, hence no containment.
        return False


def _hub_cache_repo_root(checkpoint_dir: str) -> Optional[str]:
    """Return the cache repository directory of a Hugging Face snapshot, if any.

    ``huggingface_hub`` stores one repository as ``blobs/<sha>`` next to
    ``snapshots/<revision>/<file>``, where the snapshot entries are symbolic
    links that point back into ``blobs``. A snapshot directory therefore
    resolves its shards outside itself by design, but never outside the
    ``models--<org>--<name>`` directory that holds both parts.
    """

    snapshots_dir = os.path.dirname(os.path.realpath(checkpoint_dir))
    if os.path.basename(snapshots_dir) != "snapshots":
        return None
    repo_root = os.path.dirname(snapshots_dir)
    if not os.path.basename(repo_root).startswith(_HUB_CACHE_REPO_PREFIX):
        return None
    if not os.path.isdir(os.path.join(repo_root, "blobs")):
        return None
    return repo_root


def _check_shard_name(checkpoint_dir: str, shard: str, index_path: str) -> None:
    """Reject a ``weight_map`` value that points outside the checkpoint.

    ``weight_map`` values name shard files inside the checkpoint directory. A
    value that is absolute, or that walks out of the directory with ``..``,
    makes the loader read a tensor file the caller never pointed at; the
    exported draft then ships those weights, so the index is refused instead of
    followed.

    Symbolic links are a separate case from the index value itself. A standard
    Hugging Face snapshot names plain relative shards whose targets live in the
    neighbouring ``blobs`` directory, and :func:`resolve_checkpoint_dir` returns
    exactly such a snapshot directory, so those links are supported as long as
    they stay inside the same cache repository. A link that leaves it is
    refused like any other escape.
    """

    if os.path.isabs(shard):
        raise ValueError(
            f"Checkpoint index {index_path} maps a tensor to the absolute shard "
            f"path {shard!r}; weight_map values must be shard file names inside "
            f"{checkpoint_dir}"
        )

    directory = os.path.abspath(checkpoint_dir)
    joined = os.path.normpath(os.path.join(directory, shard))
    if _inside(joined, directory):
        resolved = os.path.realpath(joined)
        if _inside(resolved, os.path.realpath(directory)):
            return
        repo_root = _hub_cache_repo_root(directory)
        if repo_root is not None and _inside(resolved, repo_root):
            return

    raise ValueError(
        f"Checkpoint index {index_path} maps a tensor to the shard {shard!r}, "
        f"which resolves outside {checkpoint_dir}; weight_map values must be "
        "shard file names inside the checkpoint directory"
    )


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
    checkpoint_dir = os.path.dirname(os.path.abspath(index_path))
    for filename in weight_map.values():
        _check_shard_name(checkpoint_dir, filename, index_path)
    return weight_map


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
            shard_tensors = _load_tensor_file(shard_path, lambda key: key in shard_keys)
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


__all__ = [
    "list_checkpoint_keys",
    "load_checkpoint_tensors",
    "read_weight_map",
    "resolve_checkpoint_dir",
]
