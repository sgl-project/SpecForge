# coding=utf-8
"""Bounded timing and NVTX ranges for the consumer input pipeline."""

from __future__ import annotations

import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Optional

import torch

_RECORDER_METADATA_KEY = "_specforge_input_pipeline_recorder"
_NVTX_PREFIX = "specforge.input_pipeline"


@dataclass
class _StageAggregate:
    count: int = 0
    total_s: float = 0.0
    maximum_s: float = 0.0

    def add(self, duration_s: float) -> None:
        self.count += 1
        self.total_s += duration_s
        self.maximum_s = max(self.maximum_s, duration_s)

    def snapshot(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "total_s": self.total_s,
            "mean_s": self.total_s / self.count if self.count else 0.0,
            "max_s": self.maximum_s,
        }


class InputPipelineRecorder:
    """Thread-safe constant-space timing aggregates shared by loader and trainer."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.perf_counter,
        emit_nvtx: bool = True,
    ) -> None:
        self._clock = clock
        self._emit_nvtx = emit_nvtx
        self._stages: dict[str, _StageAggregate] = {}
        self._counters: dict[str, int] = {}
        self._lock = threading.Lock()

    def increment(self, name: str, value: int = 1) -> None:
        if not name or isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("pipeline counter requires a name and integer value")
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def record(self, name: str, duration_s: float) -> None:
        duration_s = float(duration_s)
        if not name or not math.isfinite(duration_s) or duration_s < 0.0:
            raise ValueError("pipeline stage duration must be finite and non-negative")
        with self._lock:
            self._stages.setdefault(name, _StageAggregate()).add(duration_s)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        pushed = self._push_nvtx(name)
        started = self._clock()
        body_error: Optional[BaseException] = None
        try:
            try:
                yield
            except BaseException as exc:
                body_error = exc
                raise
        finally:
            try:
                self.record(name, self._clock() - started)
            except BaseException as timing_error:
                if body_error is None:
                    raise
                body_error.add_note(
                    f"failed to record pipeline timing: {timing_error!r}"
                )
            finally:
                if pushed:
                    try:
                        torch.cuda.nvtx.range_pop()
                    except (AttributeError, RuntimeError):
                        pass

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            stages = {
                name: aggregate.snapshot()
                for name, aggregate in sorted(self._stages.items())
            }
            counters = dict(sorted(self._counters.items()))
        return {"stages": stages, "counters": counters}

    def _push_nvtx(self, name: str) -> bool:
        if not self._emit_nvtx:
            return False
        try:
            torch.cuda.nvtx.range_push(f"{_NVTX_PREFIX}.{name}")
        except (AttributeError, RuntimeError):
            return False
        return True


def attach_input_pipeline_recorder(
    metadata: dict[str, Any], recorder: InputPipelineRecorder
) -> None:
    metadata[_RECORDER_METADATA_KEY] = recorder


def input_pipeline_recorder(
    metadata: Mapping[str, Any],
) -> Optional[InputPipelineRecorder]:
    value = metadata.get(_RECORDER_METADATA_KEY)
    return value if isinstance(value, InputPipelineRecorder) else None


@contextmanager
def batch_input_pipeline_stage(batch: Any, name: str) -> Iterator[None]:
    recorder = input_pipeline_recorder(batch.metadata)
    if recorder is None:
        yield
        return
    with recorder.stage(name):
        yield


__all__ = [
    "InputPipelineRecorder",
    "attach_input_pipeline_recorder",
    "batch_input_pipeline_stage",
    "input_pipeline_recorder",
]
