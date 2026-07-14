# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Step-scoped PyTorch profiling for the canonical training lifecycle."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProfilingOptions:
    """A profiler window expressed in completed optimizer steps."""

    enabled: bool = False
    start_step: int = 30
    num_steps: int = 4
    record_shapes: bool = False

    def __post_init__(self) -> None:
        if self.start_step < 0:
            raise ValueError("profiling start_step must be >= 0")
        if self.num_steps < 1:
            raise ValueError("profiling num_steps must be >= 1")


class StepProfiler:
    """Start/stop one profiler window without leaking it across ``fit``."""

    def __init__(
        self,
        options: ProfilingOptions,
        *,
        output_dir: str,
        trace_path: Optional[str] = None,
    ) -> None:
        self.options = options
        self.output_dir = output_dir
        self._profiler: Optional[Any] = None
        self._done = False
        self._trace_path = trace_path
        self.output_path: Optional[str] = None

    def before_micro_step(self, completed_steps: int) -> None:
        """Start before the first micro-step of the requested optimizer step."""
        if (
            not self.options.enabled
            or self._done
            or self._profiler is not None
            or completed_steps < self.options.start_step
            or completed_steps
            >= self.options.start_step + self.options.num_steps
        ):
            return

        import torch
        import torch.profiler as torch_profiler

        activities = [torch_profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch_profiler.ProfilerActivity.CUDA)
        self._profiler = torch_profiler.profile(
            activities=activities,
            with_stack=True,
            record_shapes=self.options.record_shapes,
        )
        self._profiler.start()
        logger.info("training profiler started at optimizer step %d", completed_steps)

    def after_optimizer_step(self, completed_steps: int) -> None:
        """Export after ``num_steps`` complete optimizer updates."""
        if self._profiler is None:
            return
        end_step = self.options.start_step + self.options.num_steps
        if completed_steps < end_step:
            return
        self._stop_and_export(completed_steps)

    def close(self, completed_steps: int) -> None:
        """Stop an active partial window when training exits early or fails."""
        if self._profiler is not None:
            self._stop_and_export(completed_steps)

    def _stop_and_export(self, completed_steps: int) -> None:
        profiler, self._profiler = self._profiler, None
        if profiler is None:
            return
        profiler.stop()
        os.makedirs(self.output_dir, exist_ok=True)
        rank = 0
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = int(os.environ.get("RANK", "0"))
        except (ImportError, RuntimeError, ValueError):
            rank = int(os.environ.get("RANK", "0"))
        self.output_path = self._trace_path or os.path.join(
            self.output_dir, f"profile_rank{rank}_{time.time_ns()}.trace.json.gz"
        )
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        profiler.export_chrome_trace(self.output_path)
        self._done = True
        logger.info(
            "training profiler stopped at optimizer step %d: %s",
            completed_steps,
            self.output_path,
        )


__all__ = ["ProfilingOptions", "StepProfiler"]
