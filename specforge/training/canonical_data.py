"""Unshifted capture records shared by canonical objective input preparation."""

from functools import partial

from specforge.algorithms.common.collation import (
    SequenceBatchLayout,
    pad_and_concatenate_features,
)


_SAMPLE_LOCAL_FEATURES = frozenset({"attention_mask", "position_ids", "lengths"})


def normalize_sample(raw, *, keys, max_len, loss_mask_filter=None):
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
    if loss_mask_filter is not None and not loss_mask_filter(result["loss_mask"][0]):
        raise ValueError(
            "offline record fails the objective's loss-mask requirement "
            f"({loss_mask_filter.__name__})"
        )
    if "lengths" in raw:
        lengths = raw["lengths"].flatten().long()
        if lengths.numel() == 0 or (lengths <= 0).any():
            raise ValueError("lengths must contain positive document lengths")
        starts = lengths.cumsum(0) - lengths
        result["lengths"] = lengths.minimum(length - starts)[starts < length]
    result["attention_mask"] = torch.ones_like(result["input_ids"], dtype=torch.bool)
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
    result["position_ids"] = (
        torch.arange(start, start + width, device=sample["input_ids"].device)
        .unsqueeze(0)
        .expand(sample["input_ids"].shape[0], -1)
    )
    return result


def normalize_parallel_sample(raw, *, normalizer, rank, size, overlap):
    return shard_sequence(normalizer(raw), rank=rank, size=size, overlap=overlap)


def build_normalizer(
    max_len,
    *,
    keys,
    loss_mask_filter=None,
    overlap=1,
    use_usp_preprocess=False,
    **topology,
):
    normalizer = partial(
        normalize_sample, keys=keys, max_len=max_len, loss_mask_filter=loss_mask_filter
    )
    if not use_usp_preprocess:
        return normalizer
    import torch.distributed as dist

    from specforge.distributed import get_draft_sp_group

    group = get_draft_sp_group()
    return partial(
        normalize_parallel_sample,
        normalizer=normalizer,
        rank=dist.get_rank(group),
        size=dist.get_world_size(group),
        overlap=overlap,
    )


def _sequence_keys(keys, first, *, has_attention_mask):
    return (
        *keys,
        *(("attention_mask",) if has_attention_mask else ()),
        *(("position_ids",) if "position_ids" in first else ()),
    )


def collate(features, *, keys):
    import torch

    if not features:
        raise ValueError("cannot collate an empty feature batch")
    # Streaming records already have a batch dimension; add their real length
    # mask before padding. Padding tokens must never become context keys.
    features = [
        {
            **feature,
            "attention_mask": torch.ones_like(feature["input_ids"], dtype=torch.bool),
        }
        if "attention_mask" not in feature
        else feature
        for feature in features
    ]
    required_keys = _sequence_keys(keys, features[0], has_attention_mask=True)
    result = pad_and_concatenate_features(
        features,
        sequence_axes=dict.fromkeys(required_keys, 1),
        required_keys=required_keys,
    )
    if any("lengths" in feature for feature in features):
        if len(features) != 1:
            raise ValueError("packed documents require batch size one")
        result["lengths"] = features[0]["lengths"]
    return result


def _sequence_batch_layout(refs, *, keys, diagnostic_keys):
    specs = [ref.feature_specs for ref in refs]
    if any(
        not spec
        or "lengths" in spec
        or _SAMPLE_LOCAL_FEATURES.intersection(ref.feature_keys).difference(spec)
        for ref, spec in zip(refs, specs)
    ):
        return None
    # Mixed masks and dtypes use the tensor collator's defaulting and promotion.
    masks = [spec.get("attention_mask") for spec in specs]
    if len({mask.dtype if mask is not None else None for mask in masks}) > 1:
        return None
    if len({spec["position_ids"].dtype for spec in specs if "position_ids" in spec}) > 1:
        return None
    has_attention_mask = masks[0] is not None
    required_keys = _sequence_keys(keys, specs[0], has_attention_mask=has_attention_mask)
    return SequenceBatchLayout(
        required_keys=required_keys,
        optional_keys=(),
        sequence_axes=dict.fromkeys(required_keys, 1),
        default_attention_mask=not has_attention_mask,
        diagnostic_keys=diagnostic_keys,
    )


def build_collator(*, keys, diagnostic_keys=()):
    keys = tuple(keys)
    collator = partial(collate, keys=keys)
    collator.sequence_batch_layout = partial(
        _sequence_batch_layout, keys=keys, diagnostic_keys=tuple(diagnostic_keys)
    )
    collator.sample_local_features = _SAMPLE_LOCAL_FEATURES
    return collator


def build_reader(hidden_states_path, *, strategy, keys, **kwargs):
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader

    return OfflineManifestReader(
        hidden_states_path,
        strategy=strategy,
        feature_keys=keys,
        feature_aliases={
            "hidden_states": ("aux_hidden_state",),
            "target_last_hidden_states": ("hidden_state",),
        },
        optional_features=("lengths",),
        target_repr="hidden_state" if "target_last_hidden_states" in keys else None,
        **kwargs,
    )
