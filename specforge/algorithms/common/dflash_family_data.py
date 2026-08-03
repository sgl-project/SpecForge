"""Shared DFlash-family normalization and padding adapters."""

from __future__ import annotations

from functools import partial

from specforge.algorithms.common.collation import pad_and_concatenate_features

NORMALIZER_ID = "dflash_family_offline_v1"
DSPARK_NORMALIZER_ID = "dspark_offline_v1"


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


def normalize_dspark_usp_offline_sample(
    raw,
    max_len: int,
    *,
    sp_rank: int,
    sp_size: int,
):
    """Shard all DSpark sequence tensors once per rank for offline USP.

    The feature store remains one full record per sample.  Each draft-SP rank
    slices its own sequence range at load time, including the final target state
    used by DSpark L1 and confidence supervision.  The final shard is padded to
    an equal length so HCCL all-gather has a static tensor shape.
    """
    if sp_size <= 1 or not 0 <= sp_rank < sp_size:
        raise ValueError(f"invalid DSpark USP shard rank={sp_rank}, size={sp_size}")
    import torch
    import torch.nn.functional as F

    normalized = normalize_dspark_offline_sample(raw, max_len)
    global_length = int(normalized["input_ids"].shape[1])
    chunk_length = (global_length + sp_size - 1) // sp_size
    start = sp_rank * chunk_length
    end = min(start + chunk_length, global_length)
    valid_length = max(0, end - start)

    def _slice(tensor: torch.Tensor) -> torch.Tensor:
        sliced = tensor[:, start:end]
        if valid_length < chunk_length:
            if tensor.ndim == 2:
                sliced = F.pad(sliced, (0, chunk_length - valid_length))
            elif tensor.ndim == 3:
                sliced = F.pad(sliced, (0, 0, 0, chunk_length - valid_length))
            else:  # normalize_dspark_offline_sample fixes all supported ranks.
                raise ValueError(f"unexpected DSpark USP tensor rank {tensor.ndim}")
        return sliced.contiguous()

    attention_mask = torch.zeros(
        1,
        chunk_length,
        dtype=torch.long,
        device=normalized["input_ids"].device,
    )
    attention_mask[:, :valid_length] = 1
    return {
        "input_ids": _slice(normalized["input_ids"]),
        "loss_mask": _slice(normalized["loss_mask"]),
        "hidden_states": _slice(normalized["hidden_states"]),
        "target_last_hidden_states": _slice(normalized["target_last_hidden_states"]),
        "attention_mask": attention_mask,
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


def build_dspark_offline_normalizer(max_len, *, use_usp_preprocess=False, **_topology):
    if use_usp_preprocess:
        import torch.distributed as dist

        from specforge.distributed import get_draft_sp_group

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("DSpark USP preprocessing requires process groups")
        group = get_draft_sp_group()
        if group is None:
            raise RuntimeError("DSpark USP preprocessing requires a draft SP group")
        return partial(
            normalize_dspark_usp_offline_sample,
            max_len=max_len,
            sp_rank=dist.get_rank(group),
            sp_size=dist.get_world_size(group),
        )
    return partial(normalize_dspark_offline_sample, max_len=max_len)


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
        required_keys = (
            "input_ids",
            "loss_mask",
            "hidden_states",
            "target_last_hidden_states",
        )
        has_attention_masks = ["attention_mask" in feature for feature in features]
        if any(has_attention_masks) and not all(has_attention_masks):
            raise ValueError(
                "DSpark USP batches must either all include attention_mask or none"
            )
        if all(has_attention_masks):
            required_keys = (*required_keys, "attention_mask")
        return pad_and_concatenate_features(
            features,
            sequence_axes={
                "input_ids": 1,
                "loss_mask": 1,
                "hidden_states": 1,
                "target_last_hidden_states": 1,
                "attention_mask": 1,
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
    "normalize_dspark_offline_sample",
    "normalize_dspark_usp_offline_sample",
    "normalize_offline_sample",
]
