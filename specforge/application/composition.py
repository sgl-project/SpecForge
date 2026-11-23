"""The one composition root for resolving and assembling a training run."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Sequence

from specforge.algorithms.common.providers import OfflineCaptureLayout
from specforge.algorithms.registry import AlgorithmRegistration, AlgorithmRegistry
from specforge.config import Config, flatten_drafters


@dataclass(frozen=True)
class ResolvedRun:
    """A validated config paired with its one algorithm registration.

    *name* is the drafter this run trains; a config without ``drafters`` has none.
    """

    config: Config
    algorithm: AlgorithmRegistration
    name: str | None = None


@dataclass(frozen=True)
class ResolvedOfflineCapture:
    """Algorithm-owned schema and layer plan for local feature preparation."""

    run: ResolvedRun
    draft_config: object
    capture_method: str
    capture_layers: tuple[int, ...]
    layout: OfflineCaptureLayout
    loss_mask_filter: Callable[[object], bool] | None


def bind_run(cfg: Config, algorithm: AlgorithmRegistration) -> ResolvedRun:
    """Validate a role-projected config against an existing registration."""

    from specforge.application.planning import validate_resolved_run

    validate_resolved_run(cfg, algorithm)
    return ResolvedRun(config=cfg, algorithm=algorithm)


def resolve_run(
    cfg: Config,
    registry: AlgorithmRegistry | None = None,
    *,
    target_config=None,
) -> ResolvedRun:
    """Resolve and validate all algorithm-owned behavior exactly once.

    A config with one drafter resolves as that drafter's single-draft config.
    """

    if registry is not None:
        raise ValueError(
            "Register canonical objectives instead of a combined algorithm registry"
        )
    runs = resolve_runs(cfg, target_config=target_config)
    if len(runs) > 1:
        raise ValueError(
            "resolve_run takes one run; use resolve_runs for several drafters"
        )
    return runs[0]


def resolve_runs(
    cfg: Config,
    *,
    drafter: str | None = None,
    target_config=None,
) -> tuple[ResolvedRun, ...]:
    """Resolve every run of a config: one per drafter, or the config itself.

    *drafter* keeps only that drafter's run, for the process that trains it.
    """

    from specforge.training.canonical import resolve_run as resolve_canonical_run

    named_configs = flatten_drafters(cfg)
    if drafter is not None:
        names = [name for name, _ in named_configs if name is not None]
        if drafter not in names:
            raise ValueError(f"unknown drafter {drafter!r}; this recipe has {names}")
    runs = []
    for name, run_cfg in named_configs:
        if drafter is not None and name != drafter:
            continue
        run_cfg, algorithm = resolve_canonical_run(run_cfg, target_config=target_config)
        runs.append(replace(bind_run(run_cfg, algorithm), name=name))
    return tuple(runs)


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

    if cfg.mode != "offline":
        raise ValueError("offline capture preparation requires an offline run config")

    from specforge.training.model_loading import training_target_config

    target_config = (
        training_target_config(cfg) if target_config is None else target_config
    )

    from specforge.training.model_loading import resolve_draft_config

    target = target_config
    resolved = resolve_run(cfg, registry, target_config=target)

    model_provider = resolved.algorithm.providers.model
    draft_config = resolve_draft_config(resolved.config)
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

    target_layers = target_config.get("num_hidden_layers")
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
        loss_mask_filter=model_provider.loss_mask_filter,
    )


def build_application_run(
    runs: Config | Sequence[ResolvedRun],
    registry: AlgorithmRegistry | None = None,
):
    """Build one executable process from the public config contract.

    *runs* is a config, or the resolved runs this process serves.
    """

    from specforge.training.model_loading import training_target_config

    if isinstance(runs, Config):
        target_config = training_target_config(runs)
        runs = (resolve_run(runs, registry, target_config=target_config),)
    from specforge.training.assembly import build_training_run

    return build_training_run(tuple(runs))


__all__ = [
    "ResolvedOfflineCapture",
    "ResolvedRun",
    "bind_run",
    "build_application_run",
    "resolve_offline_capture",
    "resolve_run",
    "resolve_runs",
]
