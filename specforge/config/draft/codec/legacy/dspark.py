"""Decode historical DSpark head options into explicit architecture nodes."""

from collections.abc import Mapping

from ..._validation import boolean, finite_number, positive_int
from ...architecture import DSparkSpec
from ...architecture.block import DraftOnlyBlockSpec
from ...architecture.dspark import ConfidenceHeadSpec, MarkovHeadSpec
from ...model import DraftConfig
from ..target import validate_target
from .context import _COMMON_FIELDS, _TOP_FIELDS, _check_fields, read_context
from .rules import resolve_alias

_HEAD_FIELDS = {
    "markov_rank",
    "markov_head_type",
    "enable_confidence_head",
    "confidence_head_with_markov",
    "confidence_head_alpha",
}


def decode_dspark(payload, *, target_config):
    _check_fields(
        payload,
        _TOP_FIELDS | {"markov_rank", "markov_head_type", "target_hidden_size"},
        "legacy DSpark",
    )
    method = payload.get("dflash_config")
    if method is None:
        method = {}
    if not isinstance(method, Mapping):
        raise ValueError("dflash_config must be an object or null")
    _check_fields(
        method, _COMMON_FIELDS | _HEAD_FIELDS | {"shift_label"}, "dflash_config"
    )
    if method.get("projector_type") not in (None, "dspark"):
        raise ValueError("DSpark projector_type must be dspark")
    if "shift_label" in method:
        # OnlineDSparkModel always predicts the next token at every slot.
        # This inherited Domino option was not consumed by DSpark's forward.
        boolean("shift_label", method["shift_label"])
    fields, features, block_size = read_context(payload, method, target_config)
    target_width = payload.get("target_hidden_size")
    if target_width is not None:
        positive_int("target_hidden_size", target_width)
        if target_width != fields["hidden_size"]:
            raise ValueError("legacy DSpark target_hidden_size must match hidden_size")
    for attention in fields["attention_configs"].values():
        if (
            attention.type == "gqa"
            and attention.num_key_value_heads >= attention.num_attention_heads
        ):
            raise ValueError(
                "DSpark GQA requires fewer KV heads; use attention_mode='mha' for equal counts"
            )
    rank = method.get("markov_rank")
    rank = 0 if rank is None else rank
    if type(rank) is not int or rank < 0:
        raise ValueError("markov_rank must be a nonnegative integer")
    kind = method.get("markov_head_type", "vanilla")
    if type(kind) is not str:
        raise ValueError("markov_head_type must be a string")
    kind = kind.lower()
    # Some exports redundantly write these top-level fields. The historical
    # model reads the nested fields, so disagreement must not change behavior.
    resolve_alias("markov_rank", {"markov_rank": rank}, payload)
    resolve_alias("markov_head_type", {"markov_head_type": kind}, payload)
    markov = MarkovHeadSpec(type=kind, rank=rank) if rank else None
    alpha = method.get("confidence_head_alpha")
    alpha = 0.0 if alpha is None else alpha
    finite_number("confidence_head_alpha", alpha)
    enabled = method.get("enable_confidence_head", alpha > 0)
    with_markov = method.get("confidence_head_with_markov", False)
    boolean("enable_confidence_head", enabled)
    boolean("confidence_head_with_markov", with_markov)
    if with_markov and markov is None:
        raise ValueError("confidence_head_with_markov requires a Markov head")
    config = DraftConfig(
        architecture="DSparkDraftModel",
        spec_config=DSparkSpec(
            block=DraftOnlyBlockSpec(block_size=block_size),
            target_features=features,
            markov=markov,
            confidence=ConfidenceHeadSpec(with_markov=with_markov) if enabled else None,
        ),
        **fields,
    )
    validate_target(config, target_config)
    return config
