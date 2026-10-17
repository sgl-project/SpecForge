"""EAGLE3-owned offline normalization and collation factories."""

from __future__ import annotations

from functools import partial

NORMALIZER_ID = "eagle3_offline_v1"


def normalize_offline_sample(raw, max_len: int):
    """Map the stored target states to the EAGLE3 training tensor names."""

    import torch

    hidden_state = raw["aux_hidden_state"].squeeze(0)[:max_len].unsqueeze(0)
    target = raw["hidden_state"].squeeze(0)[:max_len].unsqueeze(0)
    input_ids = raw["input_ids"][:max_len].unsqueeze(0)
    loss_mask = raw["loss_mask"][:max_len].clone().unsqueeze(0)
    if loss_mask.numel() > 0:
        loss_mask[0, -1] = 0
    return {
        "attention_mask": torch.ones_like(loss_mask, dtype=torch.long),
        "loss_mask": loss_mask,
        "target": target,
        "hidden_state": hidden_state,
        "input_ids": input_ids,
    }


def build_offline_reader(
    hidden_states_path,
    *,
    run_id,
    ttt_length,
    max_len,
):
    # Keep the heavy reader import lazy so resolving the algorithm registry is
    # side-effect free.
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader

    return OfflineManifestReader(
        hidden_states_path,
        run_id=run_id,
        strategy="eagle3",
        ttt_length=ttt_length,
        max_len=max_len,
        target_repr="hidden_state",
    )


def build_offline_normalizer(
    max_len,
    *,
    ttt_length=1,
    use_usp_preprocess=False,
):
    if not use_usp_preprocess:
        return partial(normalize_offline_sample, max_len=max_len)

    # USP sharding remains in the retained implementation until its process
    # groups become explicit provider inputs.
    import torch.distributed as dist

    from specforge.data.preprocessing import OfflineEagle3Dataset
    from specforge.distributed import get_draft_sp_group, get_sp_ring_group

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("USP preprocessing requires initialized process groups")
    sp_group = get_draft_sp_group()
    ring_group = get_sp_ring_group()
    return partial(
        OfflineEagle3Dataset.process_data_usp,
        max_len=max_len,
        ttt_length=ttt_length,
        sp_rank=dist.get_rank(sp_group),
        sp_size=dist.get_world_size(sp_group),
        ring_rank=dist.get_rank(ring_group),
        sp_ring_size=dist.get_world_size(ring_group),
    )


def build_offline_collator():
    # This retained collator owns USP-aware padding.  It can move here once the
    # distributed helper contracts are independent of specforge.data.
    from specforge.data.utils import DataCollatorWithPadding

    return DataCollatorWithPadding()


class DataCollatorWithPacking:
    """Pack one logical microbatch without changing its samples or loss weight.

    Each sample is a normalized, unpadded text feature with batch dimension one.
    The model uses ``sequence_lengths`` for attention and TTT boundaries. Keeping
    the original padded denominator makes this an execution optimization, not
    a change to the EAGLE3 objective or optimizer schedule.
    """

    def __call__(self, features):
        import torch

        if not features:
            raise ValueError("cannot pack an empty feature batch")
        keys = ("input_ids", "loss_mask", "hidden_state", "target", "attention_mask")
        lengths = []
        for feature in features:
            missing = set(keys) - feature.keys()
            if missing:
                raise KeyError(f"packed sample is missing features: {sorted(missing)}")
            ids = feature["input_ids"]
            if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
                raise ValueError("packing requires nonempty [1, length] input_ids")
            length = ids.shape[1]
            for key in keys:
                tensor = feature[key]
                ndim = 3 if key in ("hidden_state", "target") else 2
                if tensor.ndim != ndim or tensor.shape[:2] != (1, length):
                    raise ValueError(f"packing requires aligned unbatched {key}")
            if not bool((feature["attention_mask"] == 1).all()):
                raise ValueError("packing requires unpadded samples")
            if "position_ids" in feature:
                expected = torch.arange(length, device=ids.device).unsqueeze(0)
                if not torch.equal(feature["position_ids"], expected):
                    raise ValueError("packing supports standard text position_ids only")
            lengths.append(length)

        batch = {key: torch.cat([f[key] for f in features], dim=1) for key in keys}
        batch["position_ids"] = torch.cat(
            [torch.arange(n, device=batch["input_ids"].device) for n in lengths]
        ).unsqueeze(0)
        # These small descriptors stay on the host until the strategy builds
        # the device layout; no GPU scalar synchronization is needed.
        batch["sequence_lengths"] = torch.tensor(lengths, dtype=torch.long)
        batch["loss_denominator"] = torch.tensor(
            len(lengths) * max(lengths), dtype=torch.long
        )
        return batch


def build_packed_collator():
    return DataCollatorWithPacking()


def build_server_collator():
    from specforge.algorithms.common.collation import concatenate_features

    return concatenate_features


def build_padded_server_collator():
    """Accept ragged, unshifted EAGLE3 features from separate capture requests."""
    from specforge.algorithms.common.collation import pad_and_concatenate_features

    keys = ("input_ids", "attention_mask", "loss_mask", "hidden_state", "target")
    return partial(
        pad_and_concatenate_features,
        sequence_axes={key: 1 for key in keys},
        required_keys=keys,
    )


__all__ = [
    "DataCollatorWithPacking",
    "NORMALIZER_ID",
    "build_offline_collator",
    "build_offline_normalizer",
    "build_offline_reader",
    "build_packed_collator",
    "build_padded_server_collator",
    "build_server_collator",
    "normalize_offline_sample",
]
