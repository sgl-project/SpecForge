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
    result["attention_mask"] = torch.ones_like(result["input_ids"], dtype=torch.bool)
    return result


def build_normalizer(max_len, *, keys, **topology):
    return partial(normalize_sample, keys=keys, max_len=max_len)


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
    all_keys = (*keys, "attention_mask")
    return pad_and_concatenate_features(
        features, sequence_axes=dict.fromkeys(all_keys, 1), required_keys=all_keys
    )


def build_collator(*, keys):
    return partial(collate, keys=keys)


def build_reader(hidden_states_path, *, strategy, keys, **kwargs):
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader

    return OfflineManifestReader(
        hidden_states_path,
        strategy=strategy,
        feature_keys=keys,
        target_repr="hidden_state" if "target_last_hidden_states" in keys else None,
        **kwargs,
    )
