# coding=utf-8
"""One resolved contract shared by server-capture launch and production."""

from __future__ import annotations

from dataclasses import dataclass

from specforge.algorithms.registry import AlgorithmRegistration
from specforge.config import Config


@dataclass(frozen=True)
class ServerCaptureContract:
    method: str
    aux_layer_ids: tuple[int, ...]
    target_hidden_size: int
    target_vocab_size: int
    draft_vocab_size: int


def resolve_server_capture_contract(
    cfg: Config,
    *,
    algorithm: AlgorithmRegistration,
) -> ServerCaptureContract:
    """Resolve engine flags and feature dimensions from canonical model config."""
    from specforge.modeling.target.target_utils import (
        load_target_config,
        target_text_config,
        target_vocab_size,
    )
    from specforge.training.model_loading import draft_config_dict

    streaming = algorithm.providers.server_streaming_for(cfg.model.input_modality)

    target_cfg = load_target_config(
        cfg.model.target_model_path,
        cache_dir=cfg.model.cache_dir,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    target_cfg = target_text_config(target_cfg)
    model_provider = algorithm.providers.model
    draft_cfg = draft_config_dict(cfg, provider=model_provider.draft_config)
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
        target_hidden_size=int(target_cfg.hidden_size),
        target_vocab_size=target_vocab_size(target_cfg),
        draft_vocab_size=int(
            draft_cfg.get("draft_vocab_size") or draft_cfg["vocab_size"]
        ),
    )


__all__ = ["ServerCaptureContract", "resolve_server_capture_contract"]
