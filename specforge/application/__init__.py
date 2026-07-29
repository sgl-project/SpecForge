"""Application composition for the single SpecForge training entry point."""

from specforge.application.composition import (
    ResolvedOfflineCapture,
    ResolvedRun,
    bind_run,
    build_application_run,
    resolve_offline_capture,
    resolve_run,
)

__all__ = [
    "ResolvedOfflineCapture",
    "ResolvedRun",
    "bind_run",
    "build_application_run",
    "resolve_offline_capture",
    "resolve_run",
]
