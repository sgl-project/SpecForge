"""Canonical objectives, independent of launch topology and file formats."""

from . import protocols  # noqa: F401
from .base import ObjectiveModel, ObjectiveResult
from .implementations import register_implementation
from .losses import register_loss
from .plan import LossPlan, LossTerm
from .predictions import PredictionBatch, Predictions
from .projection import CorrectedProjection, LinearProjection, Projection
from .registry import (
    TrainingRequirements,
    WindowStatistic,
    build_objective_model,
    default_objective,
    default_protocol,
    objective_registrations,
    protocol_registrations,
    register_objective,
    register_protocol,
    resolve_objective,
)
from .token_kernels import register_token_kernel

__all__ = [
    "TrainingRequirements",
    "WindowStatistic",
    "ObjectiveModel",
    "ObjectiveResult",
    "build_objective_model",
    "default_objective",
    "objective_registrations",
    "register_objective",
    "register_loss",
    "register_implementation",
    "register_token_kernel",
    "register_protocol",
    "resolve_objective",
    "protocol_registrations",
    "default_protocol",
    "Predictions",
    "PredictionBatch",
    "LossTerm",
    "LossPlan",
    "Projection",
    "LinearProjection",
    "CorrectedProjection",
]
