"""Checkpoint layouts: fused parameters from per-tensor entries; join/split are inverses."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class StackedParameter:
    """A fused tensor stacked over entry index `{e}`; each entry concatenates `parts`."""

    parts: tuple[str, ...]  # per-entry tensor names, format strings taking {e}
    dim: int  # concatenation dim of the parts within one entry

    def __post_init__(self):
        if not self.parts or any(
            not isinstance(p, str) or "{e}" not in p for p in self.parts
        ):
            raise ValueError("parts must be nonempty format strings containing {e}")
        if type(self.dim) is not int or self.dim < 0:
            raise ValueError("dim must be a nonnegative integer")


@dataclass(frozen=True)
class ComponentLayout:
    """Fused local parameter names and the node field giving the entry count."""

    fused: Mapping[str, StackedParameter]
    count: str

    def __post_init__(self):
        if not self.fused or not all(
            isinstance(k, str) and isinstance(v, StackedParameter)
            for k, v in self.fused.items()
        ):
            raise ValueError("fused must map parameter names to StackedParameter")
        if not isinstance(self.count, str) or not self.count:
            raise ValueError("count must name the node field holding the entry count")

    def entries(self, config) -> int:
        count = getattr(config, self.count)
        if type(count) is not int or count <= 0:
            raise ValueError(f"{self.count} must be a positive integer")
        return count

    def part_names(self, config) -> dict[str, tuple[tuple[str, ...], ...]]:
        """Per fused parameter, the per-entry tuples of checkpoint tensor names."""
        entries = self.entries(config)
        return {
            name: tuple(
                tuple(part.format(e=e) for part in rule.parts) for e in range(entries)
            )
            for name, rule in self.fused.items()
        }


def join(layout: ComponentLayout, config, parts: Mapping[str, Tensor]) -> dict:
    """Assemble fused tensors from equal-shaped per-entry parts; every part is used once."""
    unused = set(parts)
    fused = {}
    for name, entries in layout.part_names(config).items():
        rule = layout.fused[name]
        stacked = []
        for names in entries:
            missing = [n for n in names if n not in parts]
            if missing:
                raise ValueError(f"{name}: missing checkpoint tensors {missing}")
            pieces = [parts[n] for n in names]
            # split divides an entry into equal parts, so join takes equal shapes only.
            if any(piece.shape != pieces[0].shape for piece in pieces):
                shapes = [tuple(piece.shape) for piece in pieces]
                raise ValueError(f"{name}: parts must share one shape, got {shapes}")
            unused.difference_update(names)
            stacked.append(
                pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=rule.dim)
            )
        fused[name] = torch.stack(stacked, dim=0)
    if unused:
        raise ValueError(f"unused checkpoint tensors {sorted(unused)}")
    return fused


def split(layout: ComponentLayout, config, fused: Mapping[str, Tensor]) -> dict:
    """Inverse of `join`: per-entry tensors as views of the fused ones."""
    names = layout.part_names(config)
    unknown = set(fused) - set(names)
    if unknown:
        raise ValueError(f"unknown fused parameters {sorted(unknown)}")
    parts = {}
    for name, entries in names.items():
        if name not in fused:
            raise ValueError(f"{name}: missing fused parameter")
        tensor, rule = fused[name], layout.fused[name]
        if tensor.ndim < 1 or tensor.shape[0] != len(entries):
            raise ValueError(
                f"{name}: expected {len(entries)} stacked entries, got shape "
                f"{tuple(tensor.shape)}"
            )
        for e, part_names in enumerate(entries):
            entry = tensor[e]
            if len(part_names) == 1:
                parts[part_names[0]] = entry
                continue
            if entry.ndim <= rule.dim or entry.shape[rule.dim] % len(part_names):
                raise ValueError(
                    f"{name}: dim {rule.dim} of shape {tuple(entry.shape)} does not "
                    f"split into {len(part_names)} equal parts"
                )
            width = entry.shape[rule.dim] // len(part_names)
            for part_name, piece in zip(part_names, entry.split(width, dim=rule.dim)):
                parts[part_name] = piece
    return parts
