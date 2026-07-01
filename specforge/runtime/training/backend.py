# coding=utf-8
"""E0 import shim — moved to ``specforge.training.backend``."""

from specforge.training.backend import (  # noqa: F401
    FSDPTrainingBackend,
    ParallelConfig,
    TrainingBackend,
)

__all__ = ["ParallelConfig", "TrainingBackend", "FSDPTrainingBackend"]
