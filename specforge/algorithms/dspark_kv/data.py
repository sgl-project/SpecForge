"""Offline tensors captured from an unpadded, zero-based target sequence."""

import json
from functools import partial
from pathlib import Path

from specforge.algorithms.common.collation import pad_and_concatenate_features
from specforge.algorithms.common.hidden_states_data import _normalize_hidden_states
from specforge.data.loss_mask import has_consecutive_supervised_tokens

FEATURES = ("input_ids", "loss_mask", "target_kv", "target_last_hidden_states")
NORMALIZER_ID = "dspark_target_kv_v1"


def validate_manifest(path, draft_config, target_config, target_model):
    """Require a compatible completed capture before model/optimizer assembly."""
    from specforge.modeling.draft.target_kv import capture_geometry

    root = Path(path)
    if root.is_file():
        root = root.parent
    metadata = json.loads((root / "capture-manifest.json").read_text())
    expected = {
        "format": NORMALIZER_ID,
        "target_kv_geometry": capture_geometry(draft_config, target_config),
        "state": "post_qk_norm_post_rope",
        "position_origin": 0,
        "padding": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"incompatible KV capture manifest field: {key}")
    if type(metadata.get("samples")) is not int or metadata["samples"] <= 0:
        raise ValueError(
            "KV capture manifest must describe a completed nonempty capture"
        )
    revision = getattr(target_config, "_commit_hash", None)
    if revision:
        if metadata["target_revision"] != revision:
            raise ValueError(
                "KV capture target revision disagrees with training target"
            )
    elif metadata.get("target_model") != target_model:
        raise ValueError("KV capture target model disagrees with training target")
    return metadata


def normalize_sample(raw, max_len):
    import torch

    ids, mask, kv = (raw[key] for key in FEATURES[:3])
    if ids.ndim != 1 or ids.dtype != torch.long or mask.ndim != 1:
        raise ValueError(
            "input_ids must be int64 [sequence]; loss_mask must be [sequence]"
        )
    if kv.ndim != 5 or kv.shape[2] != 2 or not kv.is_floating_point():
        raise ValueError(
            "target_kv must be floating [sequence, layers, 2, heads, head_dim]"
        )
    raw_last = raw["target_last_hidden_states"]
    if raw_last.ndim not in (2, 3) or raw_last.shape[-2] != ids.numel():
        raise ValueError(
            "DSpark final hidden must match the untruncated input sequence"
        )
    last = _normalize_hidden_states(
        raw, "target_last_hidden_states", ids.numel(), description="DSpark final hidden"
    )
    if len({ids.numel(), mask.numel(), kv.shape[0], last.shape[1]}) != 1:
        raise ValueError("DSpark KV features must have identical sequence lengths")
    if not has_consecutive_supervised_tokens(mask[:max_len]):
        raise ValueError("DSpark KV samples require two consecutive supervised tokens")
    return {
        "input_ids": ids[:max_len].unsqueeze(0),
        "loss_mask": mask[:max_len].unsqueeze(0),
        "target_kv": kv[:max_len].unsqueeze(0),
        "target_last_hidden_states": last[:, :max_len],
    }


def build_reader(hidden_states_path, *, run_id, ttt_length, max_len):
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader

    return OfflineManifestReader(
        hidden_states_path,
        strategy="dspark_kv",
        run_id=run_id,
        ttt_length=ttt_length,
        max_len=max_len,
        feature_keys=FEATURES,
        target_repr="hidden_state",
    )


def build_normalizer(max_len, **_topology):
    return partial(normalize_sample, max_len=max_len)


def build_collator():
    return partial(
        pad_and_concatenate_features,
        sequence_axes={key: 1 for key in FEATURES},
        required_keys=FEATURES,
    )
