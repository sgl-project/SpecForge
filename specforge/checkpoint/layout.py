"""Checkpoint layout resolution and weight-map containment checks."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

_CANONICAL_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
_HUB_CACHE_REPO_PREFIX = "models--"


def _inside(path: str, root: str) -> bool:
    """Return whether ``path`` is ``root`` itself or lives below it."""
    try:
        return os.path.normcase(os.path.commonpath([path, root])) == os.path.normcase(
            root
        )
    except ValueError:
        # Different drives on Windows: no common path, hence no containment.
        return False


def _hub_cache_repo_root(checkpoint_dir: str) -> str | None:
    """Return the enclosing ``models--*`` HF Hub cache root for a snapshot dir, if any."""
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
    """Reject a ``weight_map`` shard that escapes the checkpoint directory.

    Relative symlinks inside an HF Hub snapshot are allowed while their targets
    stay within the same ``models--*`` cache repository.
    """
    if os.path.isabs(shard):
        raise ValueError(
            f"Checkpoint index {index_path} maps a tensor to the absolute shard "
            f"path {shard!r}; weight_map values must be shard file names inside "
            f"{checkpoint_dir}"
        )

    directory = os.path.realpath(checkpoint_dir)
    joined = os.path.join(directory, shard)
    if _inside(os.path.normpath(joined), directory):
        # Match file lookup: follow directory symlinks before interpreting '..'.
        resolved = os.path.realpath(joined)
        if _inside(resolved, directory):
            return
        repo_root = _hub_cache_repo_root(directory)
        if repo_root is not None and _inside(resolved, repo_root):
            return

    raise ValueError(
        f"Checkpoint index {index_path} maps a tensor to the shard {shard!r}, "
        f"which leaves the checkpoint directory {checkpoint_dir}; weight_map values must be "
        "shard file names inside the checkpoint directory"
    )


def read_weight_map_file(index_path: str) -> dict[str, str]:
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
    checkpoint_dir = os.path.dirname(index_path) or os.curdir
    for filename in dict.fromkeys(weight_map.values()):
        _check_shard_name(checkpoint_dir, filename, index_path)
    return weight_map


def resolve_weight_layout(checkpoint_dir: str) -> tuple[str, str]:
    """Return ``(kind, path)`` for one deterministic checkpoint layout.

    Prefers canonical safetensors over PyTorch weights (failing on a conflicting
    single+index pair), then ``training_state.pt``, and accepts a non-canonical
    index or weight file only when unambiguous.
    """
    root = Path(checkpoint_dir)
    for name in _CANONICAL_WEIGHT_FILES:
        index = root / f"{name}.index.json"
        if (root / name).is_file() and index.is_file():
            if set(read_weight_map_file(str(index)).values()) != {name}:
                raise ValueError(
                    f"ambiguous checkpoint: {name} conflicts with {index.name}"
                )

    for name in _CANONICAL_WEIGHT_FILES:
        for kind, filename in (("single", name), ("index", f"{name}.index.json")):
            path = root / filename
            if path.is_file():
                return kind, str(path)

    # A training artifact wins over stray, non-canonical weight files.
    training_state = root / "training_state.pt"
    if training_state.is_file():
        return "single", str(training_state)

    index_files = sorted(glob.glob("*.index.json", root_dir=checkpoint_dir))
    if len(index_files) == 1:
        return "index", str(root / index_files[0])
    if len(index_files) > 1:
        raise FileNotFoundError(
            f"Multiple checkpoint index files found in {checkpoint_dir}: {index_files}"
        )

    weight_files = sorted(
        name
        for pattern in ("*.safetensors", "*.bin")
        for name in glob.glob(pattern, root_dir=checkpoint_dir)
    )
    if len(weight_files) == 1:
        return "single", str(root / weight_files[0])
    if len(weight_files) > 1:
        raise FileNotFoundError(
            f"Multiple unindexed checkpoint files found in {checkpoint_dir}: {weight_files}"
        )
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
