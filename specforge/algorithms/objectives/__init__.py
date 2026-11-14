"""Canonical objectives, independent of launch topology and file formats."""

from . import protocols  # noqa: F401
from .base import ObjectiveModel, ObjectiveResult
from .computation import LossComputation
from .implementations.registry import (
    compute_loss,
    implementation_registrations,
    register_implementation,
)
from .implementations.token_kernels import register_token_kernel
from .losses import LossOutput, loss_registrations, register_loss
from .plan import LossPlan, LossTerm
from .predictions import PredictionBatch, Predictions
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
    "loss_registrations",
    "register_implementation",
    "compute_loss",
    "implementation_registrations",
    "register_token_kernel",
    "register_protocol",
    "resolve_objective",
    "protocol_registrations",
    "default_protocol",
    "Predictions",
    "PredictionBatch",
    "LossOutput",
    "LossComputation",
    "LossTerm",
    "LossPlan",
    "Projection",
    "LinearProjection",
    "DominoSuffixProjection",
]
