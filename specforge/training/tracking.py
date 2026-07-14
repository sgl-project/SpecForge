# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Experiment tracking adapter for the canonical trainer logger seam.

``TrainerController`` deliberately depends on a tiny ``logger(metrics, step)``
callable.  This module adapts the existing W&B / TensorBoard / SwanLab / MLflow
trackers to that seam without putting tracker-specific branches in the trainer.
"""

from __future__ import annotations

import numbers
from typing import Any, Callable, Dict, Mapping, Optional


def scalar_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    """Return backend-portable scalar metrics.

    Trainer metrics are normally scalar already.  Tensor inputs are accepted so
    a custom strategy can use the same logger; non-scalar tensors/sequences are
    expanded to stable ``name/index`` keys rather than being silently dropped or
    handed to MLflow, which only accepts scalar metric values.
    """
    normalized: Dict[str, float] = {}

    def add(name: str, value: Any) -> None:
        if isinstance(value, numbers.Real):
            normalized[name] = float(value)
            return
        # Avoid importing torch when tracking is disabled or the package is used
        # for import-light configuration tooling.
        detach = getattr(value, "detach", None)
        if callable(detach):
            tensor = detach()
            if hasattr(tensor, "cpu"):
                tensor = tensor.cpu()
            numel = getattr(tensor, "numel", lambda: 0)()
            if numel == 1:
                normalized[name] = float(tensor.item())
                return
            if numel:
                for index, item in enumerate(tensor.reshape(-1).tolist()):
                    normalized[f"{name}/{index}"] = float(item)
                return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                add(f"{name}/{index}", item)

    for key, value in metrics.items():
        add(str(key), value)
    return normalized


def training_metric_names(metrics: Mapping[str, float]) -> Dict[str, float]:
    """Preserve the historical ``train/*`` tracker namespace.

    Evaluation already reports ``eval/*`` keys.  Strategy results are plain
    names because the console logger is intentionally backend-neutral; add the
    training namespace only at the external-tracker boundary.
    """
    return {
        key if key.startswith(("train/", "eval/")) else f"train/{key}": value
        for key, value in metrics.items()
    }


class TrackerLogger:
    """Callable tracker adapter with an idempotent lifecycle.

    ``close`` is intentionally separate from ``__call__`` so the single
    no-argument ``Trainer.fit`` can own cleanup through its existing
    ``on_fit_finally`` hook on success, failure, or interruption.
    """

    def __init__(
        self,
        tracker: Any,
        *,
        console_logger: Optional[Callable[[Dict[str, float], int], None]] = None,
    ) -> None:
        self.tracker = tracker
        self.console_logger = console_logger
        self._closed = False

    def __call__(self, metrics: Mapping[str, Any], step: int) -> None:
        if self._closed:
            raise RuntimeError("cannot log after TrackerLogger.close()")
        values = training_metric_names(scalar_metrics(metrics))
        if self.console_logger is not None:
            self.console_logger(values, step)
        self.tracker.log(values, step=step)

    def close(self) -> None:
        if not self._closed:
            self.tracker.close()
            self._closed = True

    def __enter__(self) -> "TrackerLogger":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def create_tracker_logger(
    args: Any,
    output_dir: str,
    *,
    console_logger: Optional[Callable[[Dict[str, float], int], None]] = None,
) -> TrackerLogger:
    """Build and validate a configured tracker behind the unified contract."""
    from specforge.tracker import get_tracker_class

    tracker_class = get_tracker_class(args.report_to)
    if tracker_class is None:
        raise ValueError(f"unsupported tracking backend: {args.report_to!r}")

    class _ValidationErrors:
        @staticmethod
        def error(message: str) -> None:
            raise ValueError(message)

    tracker_class.validate_args(_ValidationErrors(), args)
    return TrackerLogger(
        tracker_class(args, output_dir), console_logger=console_logger
    )


__all__ = [
    "TrackerLogger",
    "create_tracker_logger",
    "scalar_metrics",
    "training_metric_names",
]
