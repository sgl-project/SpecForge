"""Shared DFlash-family normalization and padding adapters."""

from __future__ import annotations

from functools import partial

from specforge.algorithms.common.collation import pad_and_concatenate_features

NORMALIZER_ID = "dflash_family_offline_v1"
DSPARK_NORMALIZER_ID = "dspark_offline_v1"


def _normalize_token_row(raw, key: str, max_len: int, *, description: str):
    tensor = raw[key]
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() != 2 or tensor.shape[0] != 1:
        raise ValueError(
            f"{description} must have shape [seq] or [1, seq], "
            f"got {tuple(tensor.shape)}"
        )
    return tensor[:, :max_len]


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

    input_ids = _normalize_token_row(
        raw,
        "input_ids",
        max_len,
        description="DFlash-family input_ids",
    )
    loss_mask = _normalize_token_row(
        raw,
        "loss_mask",
        max_len,
        description="DFlash-family loss_mask",
    )
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


def normalize_dspark_offline_sample_usp(
    raw,
    max_len: int,
    *,
    sp_rank: int,
    sp_size: int,
):
    """Normalize and sequence-shard one offline DSpark capture.

    Ulysses requires every peer to enter its all-to-all collectives with the
    same local sequence length.  The final shard is therefore zero-padded to
    ``ceil(global_length / sp_size)`` and carries an explicit validity mask.
    Absolute position IDs preserve the original sequence coordinates used by
    RoPE and by DSpark's sparse global anchor/teacher lookups.
    """

    if sp_size <= 1:
        raise ValueError("DSpark USP normalization requires sp_size > 1")
    if not 0 <= sp_rank < sp_size:
        raise ValueError(f"invalid DSpark SP rank={sp_rank}, size={sp_size}")

    import torch

    normalized = normalize_dspark_offline_sample(raw, max_len)
    global_length = int(normalized["input_ids"].shape[1])
    local_length = (global_length + sp_size - 1) // sp_size
    start = sp_rank * local_length
    stop = min(start + local_length, global_length)
    valid_length = max(stop - start, 0)

    def shard(tensor, *, sequence_axis: int):
        slices = [slice(None)] * tensor.ndim
        slices[sequence_axis] = slice(start, stop)
        local = tensor[tuple(slices)]
        if valid_length < local_length:
            shape = list(local.shape)
            shape[sequence_axis] = local_length - valid_length
            local = torch.cat([local, local.new_zeros(shape)], dim=sequence_axis)
        return local.contiguous()

    attention_mask = torch.zeros((1, local_length), dtype=torch.long)
    attention_mask[:, :valid_length] = 1
    position_ids = torch.arange(
        start,
        start + local_length,
        dtype=torch.long,
    ).unsqueeze(0)
    return {
        "input_ids": shard(normalized["input_ids"], sequence_axis=1),
        "loss_mask": shard(normalized["loss_mask"], sequence_axis=1),
        "hidden_states": shard(normalized["hidden_states"], sequence_axis=1),
        "target_last_hidden_states": shard(
            normalized["target_last_hidden_states"], sequence_axis=1
        ),
        "attention_mask": attention_mask,
        "position_ids": position_ids,
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


def build_dspark_offline_normalizer(
    max_len,
    *,
    use_usp_preprocess=False,
    **_topology,
):
    if not use_usp_preprocess:
        return partial(normalize_dspark_offline_sample, max_len=max_len)

    import torch.distributed as dist

    from specforge.distributed import get_draft_sp_group

    group = get_draft_sp_group()
    if not dist.is_available() or not dist.is_initialized() or group is None:
        raise RuntimeError(
            "DSpark USP normalizer requires initialized draft sequence parallelism"
        )
    sp_rank = dist.get_rank(group)
    sp_size = dist.get_world_size(group)
    return partial(
        normalize_dspark_offline_sample_usp,
        max_len=max_len,
        sp_rank=sp_rank,
        sp_size=sp_size,
    )


def build_dspark_streaming_transform(config):
    """Shard a server-captured DSpark sample across its online SP peers."""

    if config.training.attention_backend != "usp":
        return None

    import torch.distributed as dist

    from specforge.distributed import get_draft_sp_group

    group = get_draft_sp_group()
    if not dist.is_available() or not dist.is_initialized() or group is None:
        raise RuntimeError(
            "DSpark online USP transform requires initialized draft sequence "
            "parallelism"
        )
    return partial(
        normalize_dspark_offline_sample_usp,
        max_len=config.data.max_length,
        sp_rank=dist.get_rank(group),
        sp_size=dist.get_world_size(group),
    )


def build_collator():
    def collate(features):
        return pad_and_concatenate_features(
            features,
            sequence_axes={
                "input_ids": 1,
                "loss_mask": 1,
                "hidden_states": 1,
            },
            required_keys=("input_ids", "loss_mask", "hidden_states"),
        )

    return collate


def build_dspark_collator():
    def collate(features):
        usp_keys = ("attention_mask", "position_ids")
        usp_present = [all(key in feature for key in usp_keys) for feature in features]
        if any(usp_present) and not all(usp_present):
            raise ValueError(
                "DSpark batches cannot mix USP-sharded and unsharded samples"
            )
        required_keys = (
            "input_ids",
            "loss_mask",
            "hidden_states",
            "target_last_hidden_states",
        )
        if all(usp_present):
            required_keys += usp_keys
        return pad_and_concatenate_features(
            features,
            sequence_axes={
                "input_ids": 1,
                "loss_mask": 1,
                "hidden_states": 1,
                "target_last_hidden_states": 1,
                "attention_mask": 1,
                "position_ids": 1,
            },
            required_keys=required_keys,
        )

    return collate


__all__ = [
    "DSPARK_NORMALIZER_ID",
    "NORMALIZER_ID",
    "build_collator",
    "build_dspark_collator",
    "build_dspark_offline_normalizer",
    "build_dspark_offline_reader",
    "build_offline_normalizer",
    "build_offline_reader",
    "build_dspark_streaming_transform",
    "normalize_dspark_offline_sample",
    "normalize_dspark_offline_sample_usp",
    "normalize_offline_sample",
]
