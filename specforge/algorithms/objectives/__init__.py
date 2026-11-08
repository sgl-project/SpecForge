"""Canonical objectives, independent of launch topology and file formats."""

from . import (  # noqa: F401
    dflash,
    dflash2,
    dmtp,
    domino,
    dspark,
    dspine,
    eagle3,
    mtp,
    peagle,
)
from .base import ObjectiveModel, ObjectiveResult
from .registry import (
    build_objective_model,
    default_objective,
    objective_registrations,
    register_objective,
)

__all__ = [
    "ObjectiveModel",
    "ObjectiveResult",
    "build_objective_model",
    "default_objective",
    "objective_registrations",
    "register_objective",
]
