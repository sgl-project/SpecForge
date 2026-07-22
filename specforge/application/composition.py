"""The one composition root for resolving and assembling a training run."""

from __future__ import annotations

from dataclasses import dataclass

from specforge.algorithms.common.providers import OfflineCaptureLayout
from specforge.algorithms.registry import AlgorithmRegistration, AlgorithmRegistry
from specforge.config import Config


@dataclass(frozen=True)
class ResolvedRun:
    """A validated config paired with its one algorithm registration."""

    config: Config
    algorithm: AlgorithmRegistration


@dataclass(frozen=True)
class ResolvedOfflineCapture:
    """Algorithm-owned schema and layer plan for local feature preparation."""

    run: ResolvedRun
    draft_config: object
    capture_method: str
    capture_layers: tuple[int, ...]
    layout: OfflineCaptureLayout


def bind_run(cfg: Config, algorithm: AlgorithmRegistration) -> ResolvedRun:
    """Validate a role-projected config against an existing registration."""

    from specforge.application.planning import validate_resolved_run

    validate_resolved_run(cfg, algorithm)
    return ResolvedRun(config=cfg, algorithm=algorithm)


def resolve_run(
    cfg: Config,
    registry: AlgorithmRegistry | None = None,
) -> ResolvedRun:
    """Resolve and validate all algorithm-owned behavior exactly once."""

    if registry is None:
        from specforge.algorithms.builtin import builtin_algorithm_registry

        registry = builtin_algorithm_registry()
    try:
        algorithm = registry.resolve(cfg.training.strategy)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc

    return bind_run(cfg, algorithm)


def resolve_offline_capture(
    cfg: Config,
    *,
    target_config=None,
    registry: AlgorithmRegistry | None = None,
) -> ResolvedOfflineCapture:
    """Resolve the local producer through the same typed composition root.

    The standalone preparation command captures generic auxiliary/final target
    states. This boundary asks the selected algorithm which layers to capture
    and how those generic states must be persisted for its offline reader.
    """

    resolved = resolve_run(cfg, registry)
    if cfg.mode != "offline":
        raise ValueError("offline capture preparation requires an offline run config")

    if target_config is None:
        from transformers import AutoConfig

        target_config = AutoConfig.from_pretrained(
            cfg.model.target_model_path,
            cache_dir=cfg.model.cache_dir,
            trust_remote_code=cfg.model.trust_remote_code,
        )

    from specforge.training.model_loading import resolve_draft_config

    model_provider = resolved.algorithm.providers.model
    draft_config = resolve_draft_config(
        cfg,
        provider=model_provider.draft_config,
    )
    layers = tuple(
        model_provider.resolve_capture_layers(cfg, draft_config, target_config)
    )
    if (
        not layers
        or any(
            isinstance(layer, bool) or not isinstance(layer, int) for layer in layers
        )
        or any(layer < 0 for layer in layers)
        or len(set(layers)) != len(layers)
    ):
        raise ValueError(
            "resolved offline capture layers must be distinct non-negative "
            f"integers, got {list(layers)!r}"
        )

    text_config = getattr(target_config, "text_config", target_config)
    target_layers = getattr(text_config, "num_hidden_layers", None)
    if isinstance(target_layers, int) and any(
        layer >= target_layers for layer in layers
    ):
        raise ValueError(
            "resolved offline capture layers exceed target model depth "
            f"{target_layers}: {list(layers)!r}"
        )

    offline = resolved.algorithm.providers.offline_for(cfg.model.input_modality)
    if offline.capture_layout is None:
        raise ValueError(
            f"algorithm {resolved.algorithm.name!r} can consume offline features "
            f"for modality {cfg.model.input_modality!r} but does not register a "
            "local capture layout"
        )
    return ResolvedOfflineCapture(
        run=resolved,
        draft_config=draft_config,
        capture_method=offline.capture_layout.capture_method,
        capture_layers=layers,
        layout=offline.capture_layout,
    )


def build_application_run(
    run: Config | ResolvedRun,
    registry: AlgorithmRegistry | None = None,
):
    """Build one executable run from the public config contract."""

    resolved = run if isinstance(run, ResolvedRun) else resolve_run(run, registry)
    from specforge.training.assembly import build_training_run

    return build_training_run(
        resolved.config,
        algorithm=resolved.algorithm,
    )


__all__ = [
    "ResolvedOfflineCapture",
    "ResolvedRun",
    "bind_run",
    "build_application_run",
    "resolve_offline_capture",
    "resolve_run",
]
