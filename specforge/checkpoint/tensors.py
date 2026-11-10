"""Checkpoint metadata and selective tensor reads, without model construction."""

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from .layout import read_weight_map_file, resolve_weight_layout


@dataclass(frozen=True)
class TensorInfo:
    shape: tuple[int, ...]
    dtype: torch.dtype | str


@dataclass(frozen=True)
class TensorRef:
    store: "TensorStore"
    key: str

    @property
    def info(self):
        return self.store.metadata[self.key]

    def read(self, region=None):
        return self.store.read(self.key, region)


def _torch_state(path, *, metadata=False):
    # mmap avoids materializing unrelated tensors in legacy .bin/training files.
    state = torch.load(
        path, map_location="meta" if metadata else "cpu", weights_only=True, mmap=True
    )
    if isinstance(state, dict) and "draft_state_dict" in state:
        state = state["draft_state_dict"]
    if not isinstance(state, dict) or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
    ):
        raise ValueError(f"{path}: expected a tensor state dictionary")
    return state


class TensorStore:
    """Inspect all headers first; read only requested tensors/slices afterwards.

    Safetensors supports sliced I/O. Legacy PyTorch ZIP checkpoints are mapped
    read-only and use the restricted unpickler; arbitrary pickle objects fail.
    """

    def __init__(self, path):
        root = Path(path).expanduser()
        if root.is_file():
            layout, root = root, root.parent
        else:
            _, filename = resolve_weight_layout(str(root), allow_missing=True)
            layout = Path(filename) if filename else root / "training_state.pt"
            if not layout.is_file():
                raise FileNotFoundError(f"no checkpoint weights in {root}")
        self.root = root.resolve()
        self.metadata, self.files = {}, {}
        self._loaded = {}
        weight_map = None
        if layout.name.endswith(".index.json"):
            weight_map = read_weight_map_file(str(layout))
            filenames = sorted(set(weight_map.values()))
        else:
            filenames = [layout.name]
        for filename in filenames:
            file = self.root / filename
            if Path(filename).is_absolute() or ".." in Path(filename).parts:
                raise ValueError(f"checkpoint shard leaves its directory: {filename}")
            # HF snapshot caches legitimately link shard files to their blob store.
            if file.suffix == ".safetensors":
                with safe_open(file, framework="pt", device="cpu") as handle:
                    # get_slice reads header metadata, never the tensor payload.
                    metadata = {
                        key: TensorInfo(
                            tuple(handle.get_slice(key).get_shape()),
                            _safe_dtype(handle.get_slice(key).get_dtype()),
                        )
                        for key in handle.keys()
                    }
            else:
                metadata = {
                    k: TensorInfo(tuple(v.shape), v.dtype)
                    for k, v in _torch_state(file, metadata=True).items()
                }
            for key, info in metadata.items():
                if key in self.metadata:
                    raise ValueError(f"duplicate checkpoint tensor: {key}")
                if weight_map is not None and weight_map.get(key) != filename:
                    raise ValueError(f"index disagrees with shard for tensor {key}")
                self.metadata[key], self.files[key] = info, file
        if weight_map is not None and self.metadata.keys() != weight_map.keys():
            raise ValueError(
                f"index references missing tensors: {sorted(weight_map.keys() - self.metadata.keys())}"
            )

    def refs(self):
        return {key: TensorRef(self, key) for key in self.metadata}

    def read(self, key, region=None):
        file = self.files[key]
        if file.suffix == ".safetensors":
            with safe_open(file, framework="pt", device="cpu") as handle:
                return (
                    handle.get_tensor(key)
                    if region is None
                    else handle.get_slice(key)[region]
                )
        if file not in self._loaded:
            self._loaded[file] = _torch_state(file)
        value = self._loaded[file][key]
        return value if region is None else value[region]


def _safe_dtype(name):
    names = {
        "F64": torch.float64,
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool,
    }
    # Preserve unknown/quantized metadata; only selected weights must be loadable.
    return names.get(name, name)
