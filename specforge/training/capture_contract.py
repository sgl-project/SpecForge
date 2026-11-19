# coding=utf-8
"""One resolved contract shared by server-capture launch and production."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Sequence

from specforge.algorithms.registry import AlgorithmRegistration
from specforge.config import Config

logger = logging.getLogger(__name__)

#: Which method the servers run when drafters name different ones. SGLang's
#: DSPARK capture falls back to the DFLASH hook on targets without their own.
_METHOD_PRECEDENCE = ("dspark", "dflash", "eagle3")


@dataclass(frozen=True)
class ServerCaptureContract:
    method: str
    aux_layer_ids: tuple[int, ...]
    target_hidden_size: int
    target_vocab_size: int
    draft_vocab_size: int


def combine_capture_contracts(
    contracts: Sequence[ServerCaptureContract],
) -> ServerCaptureContract:
    """The method and layers the capture servers run to serve every drafter.

    One drafter keeps its own contract. Several capture the sorted union of
    their layers, from which the plugin cuts each drafter's own: drafters on
    layers ``(1, 9)`` and ``(1, 5)`` give servers that capture ``(1, 5, 9)``.
    The sizes of the result are the first drafter's.
    """
    first = contracts[0]
    if len(contracts) == 1:
        return first
    for contract in contracts:
        # The server concatenates layers in ascending order, so the plugin
        # can only cut a drafter's layers in that order.
        if list(contract.aux_layer_ids) != sorted(contract.aux_layer_ids):
            raise ValueError(
                "drafters sharing a capture stack need ascending target layer "
                f"ids, got {list(contract.aux_layer_ids)}"
            )
    methods = {contract.method for contract in contracts}
    unranked = sorted(methods - set(_METHOD_PRECEDENCE))
    if len(methods) > 1 and unranked:
        raise ValueError(f"cannot mix capture method {unranked} with {sorted(methods)}")
    method = next(
        (name for name in _METHOD_PRECEDENCE if name in methods), first.method
    )
    if len(methods) > 1:
        logger.warning(
            "drafters name capture methods %s; the servers run %s. The same "
            "layer id gives the same tensor on Qwen3; other targets are unchecked",
            sorted(methods),
            method,
        )
    layers = sorted(
        {layer for contract in contracts for layer in contract.aux_layer_ids}
    )
    return replace(first, method=method, aux_layer_ids=tuple(layers))


def resolve_server_capture_contract(
    cfg: Config,
    *,
    algorithm: AlgorithmRegistration,
) -> ServerCaptureContract:
    """Resolve engine flags and feature dimensions from canonical model config."""
    from specforge.training.model_loading import draft_config_dict

    streaming = algorithm.providers.server_streaming_for(cfg.model.input_modality)

    from specforge.config.draft.codec.target import bound_vocab_size
    from specforge.training.model_loading import training_target_config

    target_cfg = training_target_config(cfg)
    hidden_size, vocab_size = target_cfg["hidden_size"], bound_vocab_size(target_cfg)
    model_provider = algorithm.providers.model
    draft_cfg = draft_config_dict(cfg)
    layers = model_provider.resolve_capture_layers(cfg, draft_cfg, target_cfg)
    if not layers:
        raise ValueError("draft config does not define target capture layer ids")
    if any(
        isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
        for layer in layers
    ):
        raise ValueError(
            "resolved server capture layer ids must be non-negative integers, "
            f"got {layers!r}"
        )
    if len(set(layers)) != len(layers):
        raise ValueError(
            f"resolved server capture layer ids must be unique, got {layers!r}"
        )

    return ServerCaptureContract(
        method=streaming.capture_method,
        aux_layer_ids=tuple(layers),
        target_hidden_size=int(hidden_size),
        target_vocab_size=int(vocab_size),
        draft_vocab_size=int(
            draft_cfg.get("draft_vocab_size") or draft_cfg["vocab_size"]
        ),
    )


__all__ = [
    "ServerCaptureContract",
    "combine_capture_contracts",
    "resolve_server_capture_contract",
]
