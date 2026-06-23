import gzip
import io
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch


DEFAULT_DFLASH_CAPTURE_LAYERS = [1, 7, 12, 14, 20, 23, 24, 26, 32, 34, 39, 45]


@dataclass
class DFlashCacheManifest:
    target_model_path: str
    target_model_revision: Optional[str]
    tokenizer_path: str
    tokenizer_revision: Optional[str]
    selected_layer_ids: list[int]
    hidden_size: int
    max_length: int
    chat_template: str
    source_data_path: str
    dtype: str = "bfloat16"
    cache_format_version: int = 1


def parse_layer_ids(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, str):
        layer_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        layer_ids = [int(item) for item in value]
    if not layer_ids:
        raise ValueError("selected layer list must not be empty")
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError(f"selected layer list contains duplicates: {layer_ids}")
    if any(layer_id < 0 for layer_id in layer_ids):
        raise ValueError(f"selected layer ids must be non-negative: {layer_ids}")
    return layer_ids


def layer_stack_from_concatenated(
    hidden_states: torch.Tensor,
    *,
    num_layers: int,
    hidden_size: int,
) -> torch.Tensor:
    """Convert [seq, num_layers * hidden] or [batch, seq, ...] to layer axis."""
    expected_width = num_layers * hidden_size
    if hidden_states.shape[-1] != expected_width:
        raise ValueError(
            f"hidden width {hidden_states.shape[-1]} does not match "
            f"num_layers * hidden_size = {expected_width}"
        )
    return hidden_states.reshape(*hidden_states.shape[:-1], num_layers, hidden_size)


def cache_file_path(root: str | Path, sample_index: int, group_size: int = 2000) -> Path:
    group = sample_index // group_size
    return Path(root) / f"group_{group:06d}" / f"sample_{sample_index:012d}.ckpt"


def save_cache_record(record: dict, path: str | Path, compress: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    final_path = path.with_suffix(path.suffix + ".gz") if compress else path
    tmp_path = final_path.with_name(final_path.name + ".tmp")

    if compress:
        buffer = io.BytesIO()
        torch.save(record, buffer)
        with gzip.open(tmp_path, "wb") as f:
            f.write(buffer.getvalue())
    else:
        torch.save(record, tmp_path)
    os.replace(tmp_path, final_path)


def manifest_dict(manifest: DFlashCacheManifest) -> dict:
    return asdict(manifest)
