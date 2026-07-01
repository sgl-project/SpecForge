# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""CheckpointManager: the on-disk checkpoint lifecycle (layout, rotation, pointers).

Owns the ``output_dir`` layout ``TrainerController`` used to write inline: one
``{run_id}-step{step}/training_state.pt`` per checkpoint, plus ``latest`` and
``best`` symlinks and a ``best_meta.json``. Keeps the newest ``max_checkpoints``
(0 = keep all) and never rotates away the current ``best``. Writing is rank-0
only with a barrier, so the same call is safe under FSDP.
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Any, Dict, Optional

import torch


class CheckpointManager:
    def __init__(
        self,
        output_dir: str,
        run_id: str,
        *,
        max_checkpoints: int = 0,
        best_metric: str = "eval/simulated_acc_len",
    ) -> None:
        self.output_dir = output_dir
        self.run_id = run_id
        self.max_checkpoints = max_checkpoints
        self.best_metric = best_metric
        self.best_score: Optional[float] = None
        self.best_step: Optional[int] = None

    # -- layout ------------------------------------------------------------
    def checkpoint_dir(self, step: int) -> str:
        return os.path.join(self.output_dir, f"{self.run_id}-step{step}")

    def _state_path(self, ckpt_dir: str) -> str:
        return os.path.join(ckpt_dir, "training_state.pt")

    # -- save --------------------------------------------------------------
    def save(self, state: Dict[str, Any], step: int) -> str:
        """Persist ``state`` for ``step`` (rank 0), refresh ``latest``, rotate.

        Returns the checkpoint directory whether or not this rank wrote it, so
        every rank agrees on the resume target.
        """
        ckpt_dir = self.checkpoint_dir(step)
        if self._is_rank0():
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(state, self._state_path(ckpt_dir))
            self._point("latest", ckpt_dir)
            self._rotate()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return ckpt_dir

    def update_best(self, step: int, eval_metrics: Dict[str, float]) -> bool:
        """Track the best checkpoint by ``best_metric`` (higher is better).

        Falls back to ``avg_acc`` when the primary metric is absent (e.g. a
        strategy with no per-position acceptance). Returns True if ``step`` is the
        new best. Assumes ``step`` was already ``save``-d.
        """
        score = eval_metrics.get(self.best_metric)
        if score is None:
            score = eval_metrics.get("eval/avg_acc", eval_metrics.get("avg_acc"))
        if score is None:
            return False
        score = float(score)
        if self.best_score is not None and score <= self.best_score:
            return False
        self.best_score, self.best_step = score, step
        if self._is_rank0():
            ckpt_dir = self.checkpoint_dir(step)
            self._point("best", ckpt_dir)
            with open(os.path.join(self.output_dir, "best_meta.json"), "w") as fh:
                json.dump({"step": step, "metrics": eval_metrics}, fh, indent=2)
        return True

    # -- load --------------------------------------------------------------
    def load(self, step: Optional[int] = None, *, map_location="cpu") -> Dict[str, Any]:
        """Load a checkpoint's state dict (``latest`` when ``step`` is None)."""
        target = self.checkpoint_dir(step) if step is not None else self.latest_dir()
        if target is None:
            raise FileNotFoundError(f"no checkpoint to load under {self.output_dir}")
        return torch.load(
            self._state_path(target), map_location=map_location, weights_only=False
        )

    def latest_dir(self) -> Optional[str]:
        link = os.path.join(self.output_dir, "latest")
        if os.path.exists(link):
            return os.path.realpath(link)
        step = self._max_step_on_disk()
        return self.checkpoint_dir(step) if step is not None else None

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _is_rank0() -> bool:
        return (
            not torch.distributed.is_initialized()
        ) or torch.distributed.get_rank() == 0

    def _point(self, name: str, ckpt_dir: str) -> None:
        link = os.path.join(self.output_dir, name)
        if os.path.islink(link) or os.path.exists(link):
            try:
                os.remove(link)
            except OSError:
                return  # a non-symlink collision: leave it, the dir is source of truth
        os.symlink(os.path.abspath(ckpt_dir), link)

    def _rotate(self) -> None:
        if self.max_checkpoints <= 0:
            return
        dirs = sorted(self._all_checkpoints(), key=lambda kv: kv[0])
        for step, path in dirs[: -self.max_checkpoints]:
            if step == self.best_step:
                continue  # never rotate away the tracked best
            for f in glob.glob(os.path.join(path, "*")):
                os.remove(f)
            os.rmdir(path)

    def _all_checkpoints(self):
        pat = re.compile(rf"^{re.escape(self.run_id)}-step(\d+)$")
        for path in glob.glob(os.path.join(self.output_dir, f"{self.run_id}-step*")):
            m = pat.match(os.path.basename(path))
            if m and os.path.isdir(path):
                yield int(m.group(1)), path

    def _max_step_on_disk(self) -> Optional[int]:
        steps = [step for step, _ in self._all_checkpoints()]
        return max(steps) if steps else None


__all__ = ["CheckpointManager"]
