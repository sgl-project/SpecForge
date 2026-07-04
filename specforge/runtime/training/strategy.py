# coding=utf-8
"""Import shim — moved to ``specforge.training.strategies.base``."""

from specforge.training.strategies.base import (  # noqa: F401
    DFlashTrainStrategy,
    DominoTrainStrategy,
    DraftTrainStrategy,
    Eagle3TrainStrategy,
    StepContext,
    StepOutput,
    linear_lambda_base,
)

__all__ = [
    "DraftTrainStrategy",
    "Eagle3TrainStrategy",
    "DFlashTrainStrategy",
    "DominoTrainStrategy",
    "StepOutput",
    "StepContext",
    "linear_lambda_base",
]
