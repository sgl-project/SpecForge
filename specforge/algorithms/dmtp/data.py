"""DMTP tensors use the anchor_post_norm_v1 capture alignment."""

from functools import partial

from specforge.algorithms.common.hidden_states_data import (
    _normalize_hidden_states,
    _padded_collator,
    normalize_offline_sample,
)

FEATURES = ("input_ids", "loss_mask", "hidden_states", "mtp_hidden")
NORMALIZER_ID = "dmtp_anchor_post_norm_v1"


def normalize(raw, max_len):
    import torch

    version = raw.get("mtp_hidden_version")
    if not torch.is_tensor(version) or version.numel() != 1 or version.item() != 1:
        raise ValueError("offline DMTP features require mtp_hidden_version=1")
    result = normalize_offline_sample(raw, max_len)
    result["mtp_hidden"] = _normalize_hidden_states(
        raw, "mtp_hidden", max_len, description="DMTP mtp_hidden"
    )
    if result["mtp_hidden"].shape[1] != result["input_ids"].shape[1]:
        raise ValueError("DMTP mtp_hidden and input_ids have different lengths")
    if result["input_ids"].shape[1] < 4:
        raise ValueError("DMTP sequences need at least four tokens")
    if not loss_mask_filter(result["loss_mask"][0]):
        raise ValueError(
            "DMTP requires consecutive supervised T1 and T2 after the initial history"
        )
    return result


def build_normalizer(max_len, **_):
    return partial(normalize, max_len=max_len)


def build_collator():
    return _padded_collator(FEATURES, optional_keys=("target_last_hidden_states",))


def build_reader(hidden_states_path, *, run_id, ttt_length, max_len):
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader

    return OfflineManifestReader(
        hidden_states_path,
        run_id=run_id,
        strategy="dmtp",
        feature_keys=(*FEATURES, "mtp_hidden_version"),
        target_repr=None,
        ttt_length=ttt_length,
        max_len=max_len,
    )


def loss_mask_filter(loss_mask):
    from specforge.data.loss_mask import has_consecutive_supervised_tokens

    return has_consecutive_supervised_tokens(loss_mask[2:])
