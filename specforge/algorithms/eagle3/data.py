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


def build_server_collator():
    from specforge.algorithms.common.collation import concatenate_features

    return concatenate_features


__all__ = [
    "NORMALIZER_ID",
    "build_offline_collator",
    "build_offline_normalizer",
    "build_offline_reader",
    "build_server_collator",
    "normalize_offline_sample",
]
