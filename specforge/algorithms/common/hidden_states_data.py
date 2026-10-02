"""Shared hidden-states normalization and padding adapters.

Used by the DFlash-family (DFlash/Domino/DSpark) and MTP algorithms.
"""

from __future__ import annotations

from functools import partial

from specforge.algorithms.common.collation import pad_and_concatenate_features
from specforge.data.loss_mask import has_consecutive_supervised_tokens

NORMALIZER_ID = "dflash_family_offline_v1"
DSPARK_NORMALIZER_ID = "dspark_offline_v1"
HSPEC_NORMALIZER_ID = "hspec_offline_v1"
MTP_NORMALIZER_ID = "mtp_offline_v1"


def _normalize_hidden_states(
    raw,
    key: str,
    max_len: int,
    *,
    description: str,
):
    hidden_states = raw[key]
    if hidden_states.dim() == 3:
        if hidden_states.shape[0] != 1:
            raise ValueError(
                f"offline {description} must have shape [seq, width] or "
                f"[1, seq, width], got {tuple(hidden_states.shape)}"
            )
        hidden_states = hidden_states.squeeze(0)
    if hidden_states.dim() != 2:
        raise ValueError(
            f"offline {description} must have shape [seq, width] or "
            f"[1, seq, width], got {tuple(hidden_states.shape)}"
        )
    return hidden_states[:max_len].unsqueeze(0)


def normalize_offline_sample(raw, max_len: int):
    """Normalize raw DFlash/Domino capture tensors without target projection."""

    input_ids = raw["input_ids"][:max_len].unsqueeze(0)
    loss_mask = raw["loss_mask"][:max_len].unsqueeze(0)
    hidden_states = _normalize_hidden_states(
        raw,
        "hidden_states",
        max_len,
        description="DFlash-family hidden_states",
    )
    lengths = {
        input_ids.shape[1],
        loss_mask.shape[1],
        hidden_states.shape[1],
    }
    if len(lengths) != 1:
        raise ValueError(
            "offline DFlash-family features have mismatched sequence lengths "
            f"after truncation: input_ids={input_ids.shape[1]}, "
            f"loss_mask={loss_mask.shape[1]}, "
            f"hidden_states={hidden_states.shape[1]}"
        )
    if not has_consecutive_supervised_tokens(loss_mask[0]):
        raise ValueError(
            "offline DFlash-family samples require two consecutive supervised tokens"
        )
    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "hidden_states": hidden_states,
    }


def normalize_dspark_offline_sample(raw, max_len: int):
    """Normalize DSpark capture tensors, including target final-layer states."""

    normalized = normalize_offline_sample(raw, max_len)
    target_last_hidden_states = _normalize_hidden_states(
        raw,
        "target_last_hidden_states",
        max_len,
        description="DSpark target_last_hidden_states",
    )
    expected_length = normalized["input_ids"].shape[1]
    if target_last_hidden_states.shape[1] != expected_length:
        raise ValueError(
            "offline DSpark features have mismatched sequence lengths after "
            f"truncation: input_ids={expected_length}, "
            "target_last_hidden_states="
            f"{target_last_hidden_states.shape[1]}"
        )
    return {
        **normalized,
        "target_last_hidden_states": target_last_hidden_states,
    }


def build_offline_reader(
    strategy,
    hidden_states_path,
    *,
    run_id,
    ttt_length,
    max_len,
):
    # Transitional runtime import; the composition root will inject this port.
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader

    return OfflineManifestReader(
        hidden_states_path,
        run_id=run_id,
        strategy=strategy,
        feature_keys=("input_ids", "loss_mask", "hidden_states"),
        target_repr=None,
        ttt_length=ttt_length,
        max_len=max_len,
    )


def build_dspark_offline_reader(
    strategy,
    hidden_states_path,
    *,
    run_id,
    ttt_length,
    max_len,
):
    # Transitional runtime import; the composition root will inject this port.
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader

    return OfflineManifestReader(
        hidden_states_path,
        run_id=run_id,
        strategy=strategy,
        feature_keys=(
            "input_ids",
            "loss_mask",
            "hidden_states",
            "target_last_hidden_states",
        ),
        target_repr="hidden_state",
        ttt_length=ttt_length,
        max_len=max_len,
    )


def build_offline_normalizer(max_len, **_topology):
    return partial(normalize_offline_sample, max_len=max_len)


def build_dspark_offline_normalizer(max_len, **_topology):
    return partial(normalize_dspark_offline_sample, max_len=max_len)


def _padded_collator(required_keys, optional_keys=()):
    """Build a collator that zero-pads every listed key along the sequence axis."""

    sequence_axes = {key: 1 for key in (*required_keys, *optional_keys)}

    def collate(features):
        return pad_and_concatenate_features(
            features,
            sequence_axes=sequence_axes,
            required_keys=required_keys,
            optional_keys=optional_keys,
        )

    return collate


def build_collator():
    # The target's final hidden state rides along when the capture layout
    # carries it; offline v1 DFlash and Domino batches omit it.
    return _padded_collator(
        ("input_ids", "loss_mask", "hidden_states"),
        optional_keys=("target_last_hidden_states",),
    )


def build_dspark_collator():
    return _padded_collator(
        ("input_ids", "loss_mask", "hidden_states", "target_last_hidden_states")
    )


def normalize_hspec_offline_sample(raw, max_len: int):
    """Normalize H-Spec captures, including selected target K/V."""

    import torch

    normalized = normalize_dspark_offline_sample(raw, max_len)
    prefix_masks = raw.get("prefix_masks")
    if prefix_masks is None:
        raise ValueError(
            "offline H-Spec captures require the prefix_masks feature "
            "(1 = real context token, 0 = padding)"
        )
    prefix_masks = torch.as_tensor(prefix_masks)
    if prefix_masks.dim() == 1:
        prefix_masks = prefix_masks.unsqueeze(0)
    if prefix_masks.dim() != 2:
        raise ValueError(
            f"offline H-Spec prefix_masks must be [seq] or [1, seq], got "
            f"{tuple(prefix_masks.shape)}"
        )
    normalized["prefix_masks"] = prefix_masks[:, :max_len].bool()
    for key in ("selected_target_k", "selected_target_v"):
        tensor = raw[key]
        if tensor.dim() == 4:
            if tensor.shape[0] != 1:
                raise ValueError(
                    f"offline H-Spec {key} must have batch size 1, "
                    f"got {tuple(tensor.shape)}"
                )
            tensor = tensor.squeeze(0)
        if tensor.dim() not in (2, 3):
            raise ValueError(
                f"offline H-Spec {key} must have shape [seq, width], "
                "[1, seq, width], [seq, kv_heads, head_dim], or "
                "[1, seq, kv_heads, head_dim], got "
                f"{tuple(tensor.shape)}"
            )
        if tensor.dim() == 3:
            if tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            else:
                kv_heads, head_dim = tensor.shape[-2:]
                tensor = tensor.reshape(tensor.shape[0], kv_heads * head_dim)
        normalized[key] = tensor[:max_len].unsqueeze(0)
    sequence_length = normalized["input_ids"].shape[1]
    expected_keys = {
        "input_ids",
        "loss_mask",
        "prefix_masks",
        "hidden_states",
        "target_last_hidden_states",
        "selected_target_k",
        "selected_target_v",
    }
    if set(normalized) != expected_keys:
        raise ValueError(
            "offline H-Spec features must contain exactly "
            f"{sorted(expected_keys)}, got {sorted(normalized)}"
        )
    for key in ("selected_target_k", "selected_target_v"):
        tensor = normalized[key]
        if tensor.dim() != 3:
            raise ValueError(
                f"offline H-Spec {key} must have shape "
                "[1, seq, heads*dim*layers], "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.shape[1] != sequence_length:
            raise ValueError(
                f"offline H-Spec {key} sequence length {tensor.shape[1]} != "
                f"{sequence_length}"
            )
    prefix_masks = normalized["prefix_masks"]
    if prefix_masks.shape[1] != sequence_length:
        raise ValueError(
            "offline H-Spec prefix_masks sequence length "
            f"{prefix_masks.shape[1]} != {sequence_length}"
        )
    return normalized


def build_hspec_offline_reader(
    strategy,
    hidden_states_path,
    *,
    run_id,
    ttt_length,
    max_len,
):
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader

    return OfflineManifestReader(
        hidden_states_path,
        run_id=run_id,
        strategy=strategy,
        feature_keys=(
            "input_ids",
            "loss_mask",
            "prefix_masks",
            "hidden_states",
            "target_last_hidden_states",
            "selected_target_k",
            "selected_target_v",
        ),
        target_repr="hidden_state",
        ttt_length=ttt_length,
        max_len=max_len,
    )


def build_hspec_offline_normalizer(max_len, **_topology):
    return partial(normalize_hspec_offline_sample, max_len=max_len)


def build_hspec_collator():
    return _padded_collator(
        (
            "input_ids",
            "loss_mask",
            "prefix_masks",
            "hidden_states",
            "target_last_hidden_states",
            "selected_target_k",
            "selected_target_v",
        )
    )


def normalize_mtp_offline_sample(raw, max_len: int):
    """Normalize MTP capture tensors (no aux-layer concat, final hidden only)."""

    input_ids = raw["input_ids"][:max_len].unsqueeze(0)
    loss_mask = raw["loss_mask"][:max_len].unsqueeze(0)
    target_last_hidden_states = _normalize_hidden_states(
        raw,
        "target_last_hidden_states",
        max_len,
        description="MTP target_last_hidden_states",
    )
    lengths = {
        input_ids.shape[1],
        loss_mask.shape[1],
        target_last_hidden_states.shape[1],
    }
    if len(lengths) != 1:
        raise ValueError(
            "offline MTP features have mismatched sequence lengths after "
            f"truncation: input_ids={input_ids.shape[1]}, "
            f"loss_mask={loss_mask.shape[1]}, "
            f"target_last_hidden_states={target_last_hidden_states.shape[1]}"
        )
    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "target_last_hidden_states": target_last_hidden_states,
    }


def build_mtp_offline_reader(
    strategy,
    hidden_states_path,
    *,
    run_id,
    ttt_length,
    max_len,
):
    # Transitional runtime import; the composition root will inject this port.
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader

    return OfflineManifestReader(
        hidden_states_path,
        run_id=run_id,
        strategy=strategy,
        feature_keys=(
            "input_ids",
            "loss_mask",
            "target_last_hidden_states",
        ),
        target_repr="hidden_state",
        ttt_length=ttt_length,
        max_len=max_len,
    )


def build_mtp_offline_normalizer(max_len, **_topology):
    return partial(normalize_mtp_offline_sample, max_len=max_len)


def build_mtp_collator():
    return _padded_collator(("input_ids", "loss_mask", "target_last_hidden_states"))


__all__ = [
    "DSPARK_NORMALIZER_ID",
    "HSPEC_NORMALIZER_ID",
    "MTP_NORMALIZER_ID",
    "NORMALIZER_ID",
    "build_collator",
    "build_dspark_collator",
    "build_dspark_offline_normalizer",
    "build_dspark_offline_reader",
    "build_hspec_collator",
    "build_hspec_offline_normalizer",
    "build_hspec_offline_reader",
    "build_mtp_collator",
    "build_mtp_offline_normalizer",
    "build_mtp_offline_reader",
    "build_offline_normalizer",
    "build_offline_reader",
    "normalize_dspark_offline_sample",
    "normalize_hspec_offline_sample",
    "normalize_mtp_offline_sample",
    "normalize_offline_sample",
]
