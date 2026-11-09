# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Trainer-facing step contracts; objectives own model execution and losses."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from specforge.runtime.contracts import TrainBatch


@dataclass(frozen=True)
class StepOutput:
    """Per-step result: loss + strategy-specific metrics, kept generic so
    per-position (TTT) and single-scalar strategies share one trainer loop.

    ``loss_terms`` carries an additive objective numerator and denominator when
    gradients and reported loss must be normalized across the full optimizer
    window.
    """

    loss: torch.Tensor
    metrics: Dict[str, Any]
    ratio_metrics: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    loss_terms: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    # Additive telemetry (for example reached/accepted counts), summed across
    # the optimizer window and data-parallel ranks, never averaged.
    sum_metrics: Dict[str, Any] = field(default_factory=dict)
    # Replay a graph-free prepass once the window's local ratio totals are known.
    replay_loss: Optional[
        Callable[[Dict[str, Tuple[torch.Tensor, torch.Tensor]]], StepOutput]
    ] = None


@dataclass(frozen=True)
class StepContext:
    """Training-schedule state passed into ``forward_loss``.

    ``collect_detailed_metrics`` gates expensive diagnostics to log steps."""

    global_step: int = 0
    total_steps: Optional[int] = None
    collect_detailed_metrics: bool = True
    accumulation_steps: int = 1


class DraftTrainStrategy(abc.ABC):
    name: str
    required_features: set

    @abc.abstractmethod
    def trainable_module(self) -> nn.Module:
        """The module whose parameters the optimizer/backend owns."""

    def validate_batch(self, batch: TrainBatch) -> None:
        missing = {f for f in self.required_features if f not in batch.tensors}
        if missing:
            raise ValueError(
                f"{self.name} batch missing required features {sorted(missing)}; "
                f"present={sorted(batch.tensors)}"
            )

    @abc.abstractmethod
    def forward_loss(
        self, batch: TrainBatch, ctx: Optional["StepContext"] = None
    ) -> StepOutput: ...

    def checkpoint_state_filter(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Select the keys this strategy persists as draft weights."""
        return state_dict
