"""Unshifted capture records shared by canonical objective input preparation."""

from functools import partial


def _require_right_padding(mask):
    if (mask[:, 1:] & ~mask[:, :-1]).any():
        raise ValueError(
            "cached EAGLE attention requires contiguous context with right-padding"
        )


def normalize_sample(
    raw, *, keys, max_len, loss_mask_filter=None, require_right_padding=False
):
    import torch

    result = {}
    length = None
    optional = (
        ("attention_mask",)
        if "attention_mask" in raw and "attention_mask" not in keys
        else ()
    )
    for key in (*keys, *optional):
        if key in ("position_ids", "lengths"):
            continue  # These have independent position/document axes.
        value = raw[key]
        rank = 1 if key in ("input_ids", "loss_mask", "attention_mask") else 2
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
    if "position_ids" in raw:
        positions = raw["position_ids"][..., :max_len]
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        elif positions.ndim == 2 and positions.shape[0] == 3:
            positions = positions.unsqueeze(1)
        if tuple(positions.shape) not in ((1, length), (3, 1, length)):
            raise ValueError("position_ids must align with the capture sequence")
        result["position_ids"] = positions
    if "attention_mask" in result:
        result["attention_mask"] = result["attention_mask"].bool()
        if require_right_padding:
            _require_right_padding(result["attention_mask"])
    else:
        result["attention_mask"] = torch.ones_like(
            result["input_ids"], dtype=torch.bool
        )
    return result


def shard_sequence(sample, *, rank, size, overlap):
    """Keep owned rows plus teacher lookahead, with global RoPE coordinates."""
    import torch

    if "lengths" in sample:
        raise ValueError("packed documents are not supported with USP")
    if "attention_mask" not in sample:
        sample = {
            **sample,
            "attention_mask": torch.ones_like(sample["input_ids"], dtype=torch.bool),
        }
    _require_right_padding(sample["attention_mask"].bool())
    length = sample["input_ids"].shape[1]
    chunk = (length + size - 1) // size
    start, width = rank * chunk, chunk + overlap
    result = {}
    for name, tensor in sample.items():
        axis = tensor.ndim - 1 if name == "position_ids" else 1
        part = tensor.movedim(axis, 0)[start : start + width]
        zeros = tensor.new_zeros(width - part.shape[0], *part.shape[1:])
        result[name] = torch.cat((part, zeros)).movedim(0, axis).contiguous()
    if "position_ids" not in result:
        result["position_ids"] = (
            torch.arange(start, start + width, device=sample["input_ids"].device)
            .unsqueeze(0)
            .expand(sample["input_ids"].shape[0], -1)
        )
    # DDP/FSDP averages the rank-local gradient. Each shard gets an equal
    # share of the unsharded denominator, independent of synthetic tail rows.
    result["shard_loss_denominator"] = torch.tensor(
        sample["input_ids"].numel() / size, device=sample["input_ids"].device
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
    require_right_padding=False,
    use_usp_preprocess=False,
    **topology,
):
    normalizer = partial(
        normalize_sample,
        keys=keys,
        max_len=max_len,
        loss_mask_filter=loss_mask_filter,
        require_right_padding=require_right_padding,
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
    separate = {"position_ids", "lengths", "shard_loss_denominator"}
    all_keys = tuple(
        key for key in keys if key not in separate and key != "attention_mask"
    ) + ("attention_mask",)
    result = pad_and_concatenate_features(
        features, sequence_axes=dict.fromkeys(all_keys, 1), required_keys=all_keys
    )
    if any("position_ids" in feature for feature in features):
        positions = []
        for feature in features:
            pos = feature.get("position_ids")
            if pos is None:
                ids = feature["input_ids"]
                pos = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
            width = result["input_ids"].shape[1] - pos.shape[-1]
            positions.append(torch.nn.functional.pad(pos, (0, width)))
        if any(pos.ndim == 3 for pos in positions):
            positions = [pos.expand(3, -1, -1) for pos in positions]
        result["position_ids"] = torch.cat(positions, dim=-2)
    if "shard_loss_denominator" in features[0]:
        result["shard_loss_denominator"] = torch.stack(
            [feature["shard_loss_denominator"] for feature in features]
        ).sum()
    if any("lengths" in feature for feature in features):
        if len(features) != 1:
            raise ValueError("packed documents require batch size one")
        result["lengths"] = features[0]["lengths"]
    return result


def build_collator(*, keys):
    return partial(collate, keys=keys)


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
        optional_features=(
            ("lengths", "attention_mask", "position_ids")
            if strategy == "eagle3"
            else ("lengths",)
        ),
        target_repr="hidden_state" if "target_last_hidden_states" in keys else None,
        **kwargs,
    )
