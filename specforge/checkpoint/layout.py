"""Shared local checkpoint layout and shard-index resolution."""

import glob
import json
import os
from typing import Dict, Tuple

_CANONICAL_LAYOUTS = (
    ("single", "model.safetensors"),
    ("index", "model.safetensors.index.json"),
    ("single", "pytorch_model.bin"),
    ("index", "pytorch_model.bin.index.json"),
)


def resolve_weight_layout(
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


def read_weight_map_file(index_path: str) -> Dict[str, str]:
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
