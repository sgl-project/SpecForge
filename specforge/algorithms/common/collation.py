"""Small algorithm-owned batch collation helpers."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

PadTo = Union[int, Sequence[int]]


def concatenate_features(features):
    """Concatenate equal-shaped per-sample feature dictionaries."""

    if not features:
        raise ValueError("cannot collate an empty feature batch")
    import torch

    keys = tuple(features[0])
    expected_keys = set(keys)
    if any(set(feature) != expected_keys for feature in features[1:]):
        raise ValueError("all samples must expose the same feature keys")
    return {
        key: torch.cat([feature[key] for feature in features], dim=0) for key in keys
    }


def resolve_static_length(longest: int, pad_to) -> int:
    """Length a batch is padded to under ``training.static_shapes``.

    ``pad_to`` is one fixed length, or an ascending sequence of bucket lengths
    (``training.static_shape_buckets``): the smallest bucket that fits the
    longest sample wins. A sample longer than the (largest) length raises.
    """
    if pad_to is None:
        return int(longest)
    buckets = [int(pad_to)] if isinstance(pad_to, int) else sorted(int(b) for b in pad_to)
    for bucket in buckets:
        if longest <= bucket:
            return bucket
    raise ValueError(
        f"sample length {longest} exceeds the static batch length {buckets[-1]}"
    )


def pad_and_concatenate_features(
    features,
    *,
    sequence_axes: Mapping[str, int],
    required_keys: Sequence[str],
    optional_keys: Sequence[str] = (),
    pad_to: Optional[PadTo] = None,
):
    """Zero-pad configured tensor axes to the longest input sequence.

    ``optional_keys`` are collated when every sample carries them and omitted
    when none does; a batch that mixes both raises.  With ``pad_to`` every
    batch is padded to that fixed length instead (``training.static_shapes``),
    and a longer sample raises.
    """

    if not features:
        raise ValueError("cannot collate an empty feature batch")
    required = tuple(required_keys)
    missing = [
        (index, key)
        for index, feature in enumerate(features)
        for key in required
        if key not in feature
    ]
    if missing:
        raise KeyError(f"feature batch is missing required keys: {missing}")
    keys = list(required)
    for key in optional_keys:
        present = [key in feature for feature in features]
        if all(present):
            keys.append(key)
        elif any(present):
            raise KeyError(
                f"optional feature {key!r} must be present in every sample or "
                "omitted from every sample"
            )
    max_length = max(int(feature["input_ids"].shape[-1]) for feature in features)
    max_length = resolve_static_length(max_length, pad_to)

    import torch

    batch = {}
    for key in keys:
        axis = sequence_axes[key]
        padded = []
        for feature in features:
            tensor = feature[key]
            length = int(tensor.shape[axis])
            if length > max_length:
                raise ValueError(
                    f"feature {key!r} sequence length {length} exceeds "
                    f"input_ids length {max_length}"
                )
            if length < max_length:
                shape = list(tensor.shape)
                shape[axis] = max_length - length
                tensor = torch.cat(
                    [tensor, tensor.new_zeros(shape)],
                    dim=axis,
                )
            padded.append(tensor)
        batch[key] = torch.cat(padded, dim=0)
    return batch


__all__ = ["concatenate_features", "pad_and_concatenate_features", "resolve_static_length"]
