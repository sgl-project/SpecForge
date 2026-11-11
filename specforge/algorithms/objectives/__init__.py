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
from .losses import register_loss
from .predictions import LossTerm, PredictionBatch, Predictions
from .registry import (
    build_objective_model,
    default_objective,
    default_protocol,
    objective_registrations,
    protocol_registrations,
    register_objective,
    register_protocol,
    resolve_objective,
)

__all__ = [
    "ObjectiveModel",
    "ObjectiveResult",
    "build_objective_model",
    "default_objective",
    "objective_registrations",
    "register_objective",
    "register_loss",
    "register_protocol",
    "resolve_objective",
    "protocol_registrations",
    "default_protocol",
    "Predictions",
    "PredictionBatch",
    "LossTerm",
]
