# coding=utf-8
"""One resolved contract shared by server-capture launch and production."""

from __future__ import annotations

from dataclasses import dataclass

from specforge.config import Config

_SERVER_CAPTURE_METHODS = {
    "eagle3": "eagle3",
    "dflash": "dflash",
    "domino": "dflash",
    "dspark": "dflash",
}


@dataclass(frozen=True)
class ServerCaptureContract:
    method: str
    aux_layer_ids: tuple[int, ...]
    target_hidden_size: int
    target_vocab_size: int
    draft_vocab_size: int


def resolve_server_capture_contract(cfg: Config) -> ServerCaptureContract:
    """Resolve engine flags and feature dimensions from canonical model config."""
    from transformers import AutoConfig

    from specforge.training.model_loading import draft_config_dict
    from specforge.training.strategies.registry import resolve_strategy

    try:
        method = _SERVER_CAPTURE_METHODS[cfg.training.strategy]
    except KeyError as exc:
        raise ValueError(
            f"strategy {cfg.training.strategy!r} has no managed server-capture "
            "method"
        ) from exc

    target_cfg = AutoConfig.from_pretrained(
        cfg.model.target_model_path,
        cache_dir=cfg.model.cache_dir,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    target_cfg = getattr(target_cfg, "text_config", target_cfg)
    draft_cfg = draft_config_dict(cfg)
    strategy = resolve_strategy(cfg.training.strategy)
    layers = strategy.assembly.resolve_capture_layers(cfg, draft_cfg, target_cfg)
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
        method=method,
        aux_layer_ids=tuple(layers),
        target_hidden_size=int(target_cfg.hidden_size),
        target_vocab_size=int(target_cfg.vocab_size),
        draft_vocab_size=int(
            draft_cfg.get("draft_vocab_size") or draft_cfg["vocab_size"]
        ),
    )


__all__ = ["ServerCaptureContract", "resolve_server_capture_contract"]
