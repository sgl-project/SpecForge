"""Application composition for the single SpecForge training entry point."""

from specforge.application.composition import (
    ResolvedRun,
    bind_run,
    build_application_run,
    resolve_run,
)

__all__ = ["ResolvedRun", "bind_run", "build_application_run", "resolve_run"]
