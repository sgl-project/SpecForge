# coding=utf-8
"""Import shim — moved to ``specforge.training.controller``."""

<<<<<<<< HEAD:specforge/training/controller.py
``TrainerCore`` runs exactly one branch-free step (strategy forward/loss, backend
backward/step) plus the grad-accumulation boundary. ``TrainerController`` owns
the lifecycle: fit / evaluate / save_checkpoint. EAGLE3 and DFlash share this
unchanged — only the strategy differs.
"""

from __future__ import annotations

import itertools
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import TrainingBackend
from specforge.training.strategies.base import (
    DraftTrainStrategy,
    StepContext,
    StepOutput,
========
from specforge.training.controller import (  # noqa: F401
    Checkpoint,
    StepResult,
    TrainerController,
    TrainerCore,
>>>>>>>> 3ae106c ([DataFlow runtime] E0 — layout consolidation: runtime/ is substrate-only (move-only)):specforge/runtime/training/trainer.py
)

__all__ = ["TrainerCore", "TrainerController", "Checkpoint", "StepResult"]
