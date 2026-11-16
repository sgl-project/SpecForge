"""Final sequence-batch allocation shared by full and segmented readers.

Every returned tensor owns fresh storage for one batch. Source leases must be
released by the reader only after its copies have physically completed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class ConsumerSchedule:
    global_step: int
    accumulation_steps: int
    accumulation_remainder: int
    log_interval: int
    has_logger: bool

    def detailed_metrics(self, batch_index: int) -> bool:
        step = self.global_step + (
            self.accumulation_remainder + batch_index
        ) // self.accumulation_steps
        return self.has_logger and (step + 1) % self.log_interval == 0


def allocate_sequence_batch(refs, layout, *, device, names=None):
    """Allocate final contiguous outputs; initialize only padding, not data rows."""
    required, optional, axes = layout
    available = [set(ref.feature_specs) for ref in refs]
    for index, present in enumerate(available):
        missing = set(required) - present
        if missing:
            raise KeyError(f"sample {index} missing required features {sorted(missing)}")
    selected = list(required)
    for name in optional:
        present = [name in item for item in available]
        if any(present) and not all(present):
            raise KeyError(f"optional feature {name!r} must be present in every sample")
        if all(present):
            selected.append(name)
    if names is not None:
        selected = [name for name in selected if name in names]
    max_length = max(int(ref.feature_specs['input_ids'].shape[-1]) for ref in refs)
    output = {}
    destinations = [dict() for _ in refs]
    for name in selected:
        axis = axes[name]
        first = refs[0].feature_specs[name]
        shape = tuple(first.shape)
        if axis != 1 or not shape or shape[0] != 1:
            raise ValueError("direct sequence batching requires singleton sample axis and sequence axis 1")
        dtype = getattr(torch, first.dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"unsupported feature dtype {first.dtype!r}")
        target = torch.device(device) if dtype.is_floating_point or dtype.is_complex else torch.device('cpu')
        batch_shape = (len(refs), max_length, *shape[2:])
        batch = torch.empty(batch_shape, dtype=dtype, device=target)
        for index, ref in enumerate(refs):
            spec = ref.feature_specs[name]
            length = int(spec.shape[axis])
            if (tuple(spec.shape[:axis]) != shape[:axis]
                    or tuple(spec.shape[axis + 1:]) != shape[axis + 1:]
                    or spec.dtype != first.dtype or length > max_length):
                raise ValueError(f"incompatible direct-batch feature {name!r}")
            if length < max_length:
                batch[index:index + 1, length:].zero_()
            destinations[index][name] = batch[index:index + 1, :length]
        output[name] = batch
    return output, destinations


def copy_sample_into(store, ref, destinations: Mapping[str, torch.Tensor]):
    """Use a transport's destination API, or safely copy its CPU result."""
    method = getattr(store, 'get_into', None)
    if callable(method):
        return method(ref, destinations)
    tensors, handle = store.get(ref, device='cpu', names=list(destinations))
    try:
        for name, dest in destinations.items():
            source = tensors[name]
            if source.dtype != dest.dtype or source.shape != dest.shape:
                raise ValueError(f"source layout differs for feature {name!r}")
            dest.copy_(source, non_blocking=False)
    except BaseException:
        store.release(handle, reason='failed-batch-copy')
        raise
    return destinations, handle
