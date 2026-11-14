"""Canonical objectives, independent of launch topology and file formats."""

from . import protocols  # noqa: F401
from .base import ObjectiveModel, ObjectiveResult
from .implementations.registry import register_implementation
from .implementations.token_kernels import register_token_kernel
from .losses import LossOutput, register_loss
from .plan import LossPlan, LossTerm
from .predictions import AuxiliaryLoss, PredictionBatch, Predictions
from .projection import DominoSuffixProjection, LinearProjection, Projection
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
    "AuxiliaryLoss",
    "LossOutput",
    "LossTerm",
    "LossPlan",
    "Projection",
    "LinearProjection",
    "DominoSuffixProjection",
]
