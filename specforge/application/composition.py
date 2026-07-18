"""The one composition root for resolving and assembling a training run."""

from __future__ import annotations

from dataclasses import dataclass

from specforge.algorithms.registry import AlgorithmRegistration, AlgorithmRegistry
from specforge.config import Config


@dataclass(frozen=True)
class ResolvedRun:
    """A validated config paired with its one algorithm registration."""

    config: Config
    algorithm: AlgorithmRegistration


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


__all__ = ["ResolvedRun", "bind_run", "build_application_run", "resolve_run"]
