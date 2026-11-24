"""Draft checkpoint selection with ambiguity checks on the shared target resolver."""

from pathlib import Path

from specforge.modeling.target.checkpoint import (
    _read_weight_map_file as read_weight_map_file,
)
from specforge.modeling.target.checkpoint import (
    _resolve_weight_layout as resolve_target_weight_layout,
)


def resolve_weight_layout(checkpoint_dir):
    root = Path(checkpoint_dir)
    canonical = ("model.safetensors", "pytorch_model.bin")
    for name in canonical:
        index = root / (name + ".index.json")
        if (root / name).is_file() and index.is_file():
            if set(read_weight_map_file(str(index)).values()) != {name}:
                raise ValueError(
                    f"ambiguous checkpoint: {name} conflicts with {index.name}"
                )
    # A training artifact wins over stray, non-canonical weight files.
    if (root / "training_state.pt").is_file() and not any(
        (root / name).is_file()
        for base in canonical
        for name in (base, base + ".index.json")
    ):
        return "single", str(root / "training_state.pt")
    return resolve_target_weight_layout(checkpoint_dir)
