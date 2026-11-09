"""Unshifted capture records shared by canonical objective input preparation."""

from functools import partial


def normalize_sample(raw, *, keys, max_len):
    import torch

    result = {}
    length = None
    for key in keys:
        value = raw[key]
        rank = 1 if key in ("input_ids", "loss_mask") else 2
        if value.ndim == rank + 1 and value.shape[0] == 1:
            value = value.squeeze(0)
        if value.ndim != rank:
            raise ValueError(
                f"{key}: invalid capture tensor shape {tuple(value.shape)}"
            )
        value = value[:max_len]
        if length is not None and value.shape[0] != length:
            raise ValueError("capture tensors must have identical sequence lengths")
        length = value.shape[0]
        result[key] = value.unsqueeze(0)
    result["loss_mask"] = result["loss_mask"].clone()
    result["loss_mask"][:, -1] = 0
    result["attention_mask"] = torch.ones_like(result["input_ids"], dtype=torch.bool)
    if "lengths" in raw:
        lengths = raw["lengths"].flatten().to(dtype=torch.long)
        if lengths.numel() == 0 or (lengths <= 0).any():
            raise ValueError("lengths must contain positive document lengths")
        starts = lengths.cumsum(0) - lengths
        result["lengths"] = (lengths.minimum(length - starts))[starts < length]
    return result


def shard_sequence(sample, *, rank, size, overlap):
    """Keep owned rows plus teacher lookahead, with global RoPE coordinates."""
    import torch

    if "lengths" in sample:
        raise ValueError("packed documents are not supported with USP")
    length = sample["input_ids"].shape[1]
    chunk = (length + size - 1) // size
    start, width = rank * chunk, chunk + overlap
    result = {}
    for name, tensor in sample.items():
        part = tensor[:, start : start + width]
        zeros = tensor.new_zeros(
            tensor.shape[0], width - part.shape[1], *tensor.shape[2:]
        )
        result[name] = torch.cat((part, zeros), dim=1).contiguous()
    result["position_ids"] = torch.arange(start, start + chunk).unsqueeze(0)
    return result


def normalize_parallel_sample(raw, *, keys, max_len, rank, size, overlap):
    sample = normalize_sample(raw, keys=keys, max_len=max_len)
    return shard_sequence(sample, rank=rank, size=size, overlap=overlap)


def build_normalizer(
    max_len, *, keys, ttt_length=1, use_usp_preprocess=False, **topology
):
    if not use_usp_preprocess:
        return partial(normalize_sample, keys=keys, max_len=max_len)
    import torch.distributed as dist

    from specforge.distributed import get_draft_sp_group

    group = get_draft_sp_group()
    return partial(
        normalize_parallel_sample,
        keys=keys,
        max_len=max_len,
        rank=dist.get_rank(group),
        size=dist.get_world_size(group),
        overlap=ttt_length,
    )


def collate(features, *, keys):
    import torch

    from specforge.algorithms.common.collation import pad_and_concatenate_features

    # Streaming records already have a batch dimension; add their real length
    # mask before padding. Padding tokens must never become context keys.
    features = [
        {
            **feature,
            "attention_mask": feature.get(
                "attention_mask",
                torch.ones_like(feature["input_ids"], dtype=torch.bool),
            ),
        }
        for feature in features
    ]
    positions = ("position_ids",) if "position_ids" in features[0] else ()
    all_keys = (*keys, "attention_mask", *positions)
    result = pad_and_concatenate_features(
        features, sequence_axes=dict.fromkeys(all_keys, 1), required_keys=all_keys
    )
    if any("lengths" in feature for feature in features):
        if len(features) != 1:
            raise ValueError("packed documents require batch size one")
        result["lengths"] = features[0]["lengths"].flatten()
    return result


def build_collator(*, keys):
    return partial(collate, keys=keys)


def build_reader(hidden_states_path, *, strategy, keys, **kwargs):
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader

    return OfflineManifestReader(
        hidden_states_path,
        strategy=strategy,
        feature_keys=keys,
        feature_aliases=(
            {
                "hidden_states": ("aux_hidden_state",),
                "target_last_hidden_states": ("hidden_state",),
            }
            if strategy in ("eagle3", "peagle")
            else {}
        ),
        optional_features=("lengths",),
        target_repr="hidden_state" if "target_last_hidden_states" in keys else None,
        **kwargs,
    )
