# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""CheckpointManager: the on-disk checkpoint lifecycle (layout, rotation, pointers).

Owns the ``output_dir`` layout ``TrainerController`` used to write inline. Each
checkpoint is one ``{run_id}-step{step}/`` directory holding:

* ``training_state.pt`` — the rank-0 **shared** payload (export-filtered draft
  weights, ``global_step`` / ``epoch`` / ``epoch_batch`` counters, ``strategy``,
  ``run_id``, ``world_size``);
* ``training_state_rank{r}.pt`` — one per rank, the **rank-local** payload
  (optimizer/scheduler state and RNG — under FSDP ``use_orig_params`` the AdamW
  moments live on each rank's shard views, so persisting only rank0's copy would
  corrupt every other rank on resume).

Plus ``latest`` and ``best`` symlinks and a ``best_meta.json`` in ``output_dir``.
Keeps the newest ``max_checkpoints`` (0 = keep all) and never rotates away the
tracked ``best``; the best score/step are re-read from ``best_meta.json`` on
construction so best protection survives a restart. Multi-rank runs require
``output_dir`` on a filesystem every rank can read and write — the same
requirement resuming the per-rank files already imposes.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
from typing import Any, Dict, Optional

import torch

STATE_FILE = "training_state.pt"


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
        self._load_best_meta()

    # -- layout ------------------------------------------------------------
    def checkpoint_dir(self, step: int) -> str:
        return os.path.join(self.output_dir, f"{self.run_id}-step{step}")

    def _state_path(self, ckpt_dir: str) -> str:
        return os.path.join(ckpt_dir, STATE_FILE)

    @staticmethod
    def _rank_file(rank: int) -> str:
        return f"training_state_rank{rank}.pt"

    # -- save --------------------------------------------------------------
    def save(
        self, state: Optional[Dict[str, Any]], step: int, *, rank_state=None
    ) -> str:
        """Persist a checkpoint for ``step``; refresh ``latest``; rotate.

        ``state`` is the shared payload, written by rank 0 only (other ranks may
        pass ``None``). ``rank_state`` — when given — is written by **every**
        rank to its own ``training_state_rank{r}.pt`` (optimizer/RNG are
        rank-local). Returns the checkpoint directory on every rank, so all
        ranks agree on the resume target.
        """
        ckpt_dir = self.checkpoint_dir(step)
        if self.is_rank0():
            os.makedirs(ckpt_dir, exist_ok=True)
        self._barrier()  # the dir exists before any rank writes into it
        if rank_state is not None:
            torch.save(rank_state, os.path.join(ckpt_dir, self._rank_file(self._rank())))
        self._barrier()  # every rank file is on disk before `latest` moves
        if self.is_rank0():
            if state is not None:
                torch.save(state, self._state_path(ckpt_dir))
            self._point("latest", ckpt_dir)
            self._rotate()
        self._barrier()  # no rank proceeds before the checkpoint is complete
        return ckpt_dir

    # -- best tracking -----------------------------------------------------
    def score(self, eval_metrics: Dict[str, Any]) -> Optional[float]:
        """Extract the best-tracking score from an eval-metrics dict.

        Falls back to ``avg_acc`` when the primary metric is absent (e.g. a
        strategy with no per-position acceptance); ``None`` when neither exists.
        """
        s = eval_metrics.get(self.best_metric)
        if s is None:
            s = eval_metrics.get("eval/avg_acc", eval_metrics.get("avg_acc"))
        return float(s) if s is not None else None

    def is_better(self, eval_metrics: Dict[str, Any]) -> bool:
        """True when these metrics beat the tracked best (higher is better)."""
        s = self.score(eval_metrics)
        return s is not None and (self.best_score is None or s > self.best_score)

    def update_best(
        self, step: int, eval_metrics: Dict[str, Any], *, force: bool = False
    ) -> bool:
        """Track ``step`` as the best checkpoint if its metrics beat the record.

        Assumes ``step`` was already ``save``-d (the ``best`` pointer needs a
        directory to point at). Persists ``best_meta.json`` so the record — and
        rotation protection — survive a restart. ``force=True`` records
        unconditionally: used after a rank-agreed (broadcast) verdict, where a
        rank whose stale local record disagrees must still follow rank0.
        Returns True on a new best.
        """
        score = self.score(eval_metrics)
        if score is None or (not force and not self.is_better(eval_metrics)):
            return False
        self.best_score, self.best_step = score, step
        if self.is_rank0():
            ckpt_dir = self.checkpoint_dir(step)
            self._point("best", ckpt_dir)
            with open(os.path.join(self.output_dir, "best_meta.json"), "w") as fh:
                json.dump(
                    {
                        "step": step,
                        "score": self.best_score,
                        "metric": self.best_metric,
                        "metrics": eval_metrics,
                    },
                    fh,
                    indent=2,
                )
        return True

    def _load_best_meta(self) -> None:
        # Rehydrate best_score/best_step so a resumed process neither rotates
        # away the on-disk best nor lets a worse score overwrite it.
        try:
            with open(os.path.join(self.output_dir, "best_meta.json")) as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            return  # no meta yet, or unreadable: start fresh
        score = meta.get("score")
        if score is None and isinstance(meta.get("metrics"), dict):
            score = meta["metrics"].get(self.best_metric)
        step = meta.get("step")
        if score is None or step is None:
            return
        if not os.path.isdir(self.checkpoint_dir(int(step))):
            return  # the recorded best no longer exists on disk
        self.best_score, self.best_step = float(score), int(step)

    # -- load --------------------------------------------------------------
    def load(self, step: Optional[int] = None, *, map_location="cpu") -> Dict[str, Any]:
        """Load a checkpoint's shared state dict (``latest`` when ``step`` is None)."""
        target = self.checkpoint_dir(step) if step is not None else self.latest_dir()
        if target is None:
            raise FileNotFoundError(f"no checkpoint to load under {self.output_dir}")
        return torch.load(
            self._state_path(target), map_location=map_location, weights_only=False
        )

    @classmethod
    def read_resume_state(cls, path_or_uri: str, *, map_location="cpu") -> Dict[str, Any]:
        """Read a checkpoint into the resume-state dict for **this rank**.

        Accepts a checkpoint directory, its ``training_state.pt``, or a
        ``file://`` URI. Merges the shared payload with this rank's
        ``training_state_rank{r}.pt`` (as ``optimizer_state_dict`` /
        ``rng_state``). The per-rank optimizer/RNG shards only make sense at the
        world size that wrote them, so a world-size mismatch fails fast instead
        of silently corrupting optimizer moments.
        """
        path = str(path_or_uri)
        if path.startswith("file://"):
            path = path[len("file://") :]
        if os.path.basename(path) == STATE_FILE:
            path = os.path.dirname(path)
        state = torch.load(
            os.path.join(path, STATE_FILE),
            map_location=map_location,
            weights_only=False,
        )
        initialized = torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if initialized else 0
        world = torch.distributed.get_world_size() if initialized else 1
        rank_path = os.path.join(path, cls._rank_file(rank))
        if os.path.exists(rank_path):
            saved_world = state.get("world_size")
            if saved_world is not None and saved_world != world:
                raise ValueError(
                    f"checkpoint {path} was written at world_size={saved_world} but "
                    f"resume is running at world_size={world}: per-rank optimizer/RNG "
                    f"shards do not transfer across world sizes — resume at the "
                    f"original world size (or load the draft weights only)"
                )
            rank_state = torch.load(
                rank_path, map_location=map_location, weights_only=False
            )
            state["optimizer_state_dict"] = rank_state.get("optimizer")
            state["rng_state"] = rank_state.get("rng")
        elif glob.glob(os.path.join(path, "training_state_rank*.pt")):
            raise ValueError(
                f"checkpoint {path} has per-rank state files but none for rank "
                f"{rank}: it was written at world_size={state.get('world_size')} — "
                f"resume at that world size"
            )
        elif world > 1 and ("optimizer_state_dict" in state or "rng_state" in state):
            raise ValueError(
                f"checkpoint {path} uses the legacy single-file layout: its "
                f"optimizer/RNG state is rank0's only and cannot seed "
                f"world_size={world} — resume it at world_size==1"
            )
        # else: weights-only or legacy single-rank checkpoint; use it as-is.
        return state

    def latest_dir(self) -> Optional[str]:
        link = os.path.join(self.output_dir, "latest")
        if os.path.exists(link):
            return os.path.realpath(link)
        step = self._max_step_on_disk()
        return self.checkpoint_dir(step) if step is not None else None

    # -- internals ---------------------------------------------------------
    @staticmethod
    def is_rank0() -> bool:
        return (
            not torch.distributed.is_initialized()
        ) or torch.distributed.get_rank() == 0

    @staticmethod
    def _rank() -> int:
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    @staticmethod
    def _barrier() -> None:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _point(self, name: str, ckpt_dir: str) -> None:
        link = os.path.join(self.output_dir, name)
        if os.path.islink(link) or os.path.exists(link):
            try:
                os.remove(link)
            except OSError:
                return  # a non-symlink collision: leave it, the dir is source of truth
        try:
            os.symlink(os.path.abspath(ckpt_dir), link)
        except OSError:
            # e.g. a filesystem without symlink support: the step directories
            # (+ best_meta.json) remain the source of truth; latest_dir() falls
            # back to the max step on disk.
            return

    def _rotate(self) -> None:
        if self.max_checkpoints <= 0:
            return
        dirs = sorted(self._all_checkpoints(), key=lambda kv: kv[0])
        for step, path in dirs[: -self.max_checkpoints]:
            if step == self.best_step:
                continue  # never rotate away the tracked best
            shutil.rmtree(path)

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
