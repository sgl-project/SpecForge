"""Resolve legacy Domino head dimensions and alignment into typed nodes."""

from collections.abc import Mapping

from ...architecture import DominoSpec
from ...architecture.domino import DominoBlockSpec, GRUHeadSpec
from ...model import DraftConfig
from .context import _COMMON_FIELDS, _TOP_FIELDS, _check_fields, read_context
from .rules import resolve_alias

_HEAD_FIELDS = {"emb_dim", "gru_hidden_dim", "pure_draft_prefix_len", "shift_label"}


def decode_domino(payload, *, target_config):
    _check_fields(payload, _TOP_FIELDS | {"emb_dim"}, "legacy Domino")
    method = payload.get("dflash_config")
    if not isinstance(method, Mapping):
        raise ValueError("Domino requires dflash_config with GRU dimensions")
    _check_fields(method, _COMMON_FIELDS | _HEAD_FIELDS, "dflash_config")
    if method.get("projector_type") not in (None, "domino"):
        raise ValueError("Domino projector_type must be domino")
    # The old constructor reads nested dimensions. A redundant top-level
    # emb_dim must agree; it cannot fill a missing effective nested setting.
    projection_size = resolve_alias("emb_dim", method)
    resolve_alias("emb_dim", method, payload)
    hidden_size = resolve_alias("gru_hidden_dim", method)
    fields, features, block_size = read_context(payload, method, target_config)
    config = DraftConfig(
        architecture="DominoDraftModel",
        spec_config=DominoSpec(
            block=DominoBlockSpec(
                block_size=block_size, shift_label=method.get("shift_label", False)
            ),
            target_features=features,
            gru=GRUHeadSpec(hidden_size=hidden_size, projection_size=projection_size),
            pure_draft_prefix_len=method.get("pure_draft_prefix_len", 0),
        ),
        **fields,
    )
    return config
