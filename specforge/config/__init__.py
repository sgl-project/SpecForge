# coding=utf-8
"""Typed run configuration (Pydantic) + loader for the ``specforge`` CLI."""

__all__ = [
    "Config",
    "ModelConfig",
    "DataConfig",
    "DeploymentConfig",
    "DisaggregatedDeploymentConfig",
    "ManagedLocalMooncakeConfig",
    "ManagedLocalCaptureServerConfig",
    "ManagedLocalStackConfig",
    "TrainingConfig",
    "TrackingConfig",
    "ProfilingConfig",
    "RuntimeConfig",
    "SGLANG_CAPTURE_CONTEXT_HEADROOM",
    "TrainerDeploymentConfig",
    "load_config",
    "apply_overrides",
]


def __getattr__(name):
    # Importing draft specs must not load the run schema or runtime packages.
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from specforge.config import schema

    value = getattr(schema, name)
    globals()[name] = value
    return value
