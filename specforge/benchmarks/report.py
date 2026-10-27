# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Result records, the console summary, and the JSON report."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskMetrics:
    """Aggregate measurements for one task under one server configuration."""

    num_samples: int
    #: Requests in the timed phase; exceeds ``num_samples`` for multi-turn tasks.
    num_requests: int
    output_tokens: int
    latency_seconds: float
    throughput_tokens_per_second: float
    #: ``output_tokens / spec_verify_count``; ``None`` without speculation.
    accept_length: Optional[float] = None
    spec_verify_count: Optional[int] = None
    #: Fraction of scored samples that were correct; ``None`` for unscored tasks.
    accuracy: Optional[float] = None
    num_scored: int = 0


@dataclass
class RunRecord:
    """One row of the report: which config, which task, what was measured."""

    #: ``None`` when measuring a running server whose configuration is unknown.
    config: Optional[Dict[str, Any]]
    task: Dict[str, Any]
    metrics: TaskMetrics

    @property
    def config_label(self) -> str:
        return self.config["label"] if self.config else "server"


@dataclass
class BenchmarkReport:
    model: str
    draft_model: Optional[str]
    sampling: Dict[str, Any]
    concurrency: int
    created_at: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z")
    )
    runs: List[RunRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write(self, output_dir: str, name: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"{name}_{stamp}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path


_COLUMNS = (
    ("config", "<", 18),
    ("task", "<", 14),
    ("samples", ">", 7),
    ("tokens", ">", 9),
    ("tok/s", ">", 9),
    ("accept", ">", 7),
    ("accuracy", ">", 9),
)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_summary(report: BenchmarkReport) -> str:
    """Render the runs as a fixed-width table."""
    header = "  ".join(f"{name:{align}{width}}" for name, align, width in _COLUMNS)
    rule = "-" * len(header)
    lines = [header, rule]
    for run in report.runs:
        metrics = run.metrics
        values = (
            run.config_label,
            run.task["name"],
            metrics.num_samples,
            metrics.output_tokens,
            _fmt(metrics.throughput_tokens_per_second, 1),
            _fmt(metrics.accept_length, 3),
            (
                _fmt(metrics.accuracy * 100, 1) + "%"
                if metrics.accuracy is not None
                else "-"
            ),
        )
        lines.append(
            "  ".join(
                f"{str(value):{align}{width}}"
                for value, (_, align, width) in zip(values, _COLUMNS)
            )
        )
    return "\n".join(lines)


__all__ = ["BenchmarkReport", "RunRecord", "TaskMetrics", "format_summary"]
