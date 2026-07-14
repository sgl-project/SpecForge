# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""On-disk checkpoint lifecycle: layout, rotation, latest pointer, resume reads.

A checkpoint is ``{run_id}-step{N}/`` under ``output_dir``: ``training_state.pt``
(rank0 shared payload) plus ``training_state_rank{r}.pt`` per rank (optimizer/RNG
are rank-local under FSDP). A run-scoped ``{run_id}-latest`` link sits beside
them; multi-rank runs need a shared filesystem.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import shutil
from typing import Any, Dict, Iterator, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

STATE_FILE = "training_state.pt"


class CheckpointManager:
    def __init__(
        self,
        output_dir: str,
        run_id: str,
        *,
        max_checkpoints: int = 0,
    ) -> None:
        self.output_dir = output_dir
        self.run_id = run_id
        self.max_checkpoints = max_checkpoints

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
        self,
        state: Optional[Dict[str, Any]],
        step: int,
        *,
        rank_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Write ``step``'s checkpoint, first deleting any on-disk steps >= step
        (fork/rollback semantics), then repoint ``{run_id}-latest`` and rotate.
        ``state`` is the shared payload (rank0 writes it); ``rank_state`` is
        written by every rank. Collective: any rank's failure raises on all ranks.
        """
        ckpt_dir = self.checkpoint_dir(step)
        err = ""
        try:
            if self.is_rank0():
                self._rewind(step)
        except Exception as exc:
            err = f"rewind failed: {type(exc).__name__}: {exc}"
        self._barrier()  # stale >= step dirs are gone before any rank recreates them
        if not err:
            try:
                os.makedirs(ckpt_dir, exist_ok=True)
                if rank_state is not None:
                    self._atomic_save(
                        rank_state,
                        os.path.join(ckpt_dir, self._rank_file(self._rank())),
                    )
                if self.is_rank0() and state is not None:
                    self._atomic_save(state, self._state_path(ckpt_dir))
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
        self._all_ok(err)
        if self.is_rank0():
            self._point(f"{self.run_id}-latest", ckpt_dir)
            self._rotate(keep_step=step)
        self._barrier()  # no rank proceeds before the checkpoint is complete
        return ckpt_dir

    def _rewind(self, step: int) -> None:
        # Fork semantics: saving step S invalidates on-disk steps >= S.
        stale = sorted(path for s, path in self._step_dirs() if s >= step)
        if stale:
            logger.warning(
                "rewinding checkpoint timeline at step %d: deleting %s",
                step,
                ", ".join(stale),
            )
            for path in stale:
                shutil.rmtree(path)

    def _all_ok(self, err: str) -> None:
        # Barrier replacement: every rank learns every rank's outcome, so one
        # rank's FS failure raises everywhere instead of stranding the group.
        if torch.distributed.is_initialized():
            errs = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(errs, err)
        else:
            errs = [err]
        failures = [(r, e) for r, e in enumerate(errs) if e]
        if failures:
            raise RuntimeError(
                "checkpoint save failed: "
                + "; ".join(f"rank {r}: {e}" for r, e in failures)
            )

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
    def read_resume_state(
        cls,
        path: str,
        *,
        map_location="cpu",
        require_full_state: bool = True,
    ) -> Dict[str, Any]:
        """Read a checkpoint into this rank's resume dict: the shared payload plus
        ``'backend'`` = this rank's ``training_state_rank{r}.pt`` content, passed
        through untouched (``{}`` when absent and ``require_full_state`` is False).
        Accepts a checkpoint dir, its ``training_state.pt``, or a ``file://`` URI.
        """
        path = str(path)
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
        saved_world = state.get("world_size")
        rank_path = os.path.join(path, cls._rank_file(rank))
        if os.path.exists(rank_path) and (saved_world is None or saved_world == world):
            state["backend"] = torch.load(
                rank_path, map_location=map_location, weights_only=False
            )
            return state
        state["backend"] = {}
        if require_full_state and int(state.get("global_step") or 0) > 0:
            raise ValueError(
                f"checkpoint {path} has no usable per-rank state for rank {rank} at "
                f"world_size={world} (written at world_size={saved_world}): resuming "
                f"training needs each rank's optimizer/RNG shard — resume at the "
                f"original world size, or pass require_full_state=False to restore "
                f"weights and counters only"
            )
        return state

    def latest_dir(self) -> Optional[str]:
        """Directory of the newest complete checkpoint, or None."""
        link = os.path.join(self.output_dir, f"{self.run_id}-latest")
        if os.path.islink(link):
            target = os.path.realpath(link)
            if os.path.isfile(self._state_path(target)):
                return target
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

    @staticmethod
    def _atomic_save(obj: Any, path: str) -> None:
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def _point(self, name: str, ckpt_dir: str) -> None:
        link = os.path.join(self.output_dir, name)
        try:
            if os.path.islink(link):
                os.remove(link)
            elif os.path.isdir(link):
                logger.warning("removing non-symlink %s shadowing checkpoints", link)
                shutil.rmtree(link)
            elif os.path.exists(link):
                logger.warning("removing non-symlink %s shadowing checkpoints", link)
                os.remove(link)
            # Relative target survives relocating output_dir to another mount.
            os.symlink(os.path.basename(ckpt_dir), link)
        except OSError:
            return  # no symlink support: complete step dirs stay authoritative

    def _rotate(self, keep_step: int) -> None:
        if self.max_checkpoints <= 0:
            return
        dirs = sorted(self._all_checkpoints())
        for step, path in dirs[: -self.max_checkpoints]:
            if step == keep_step:
                continue
            try:
                shutil.rmtree(path)
            except OSError as exc:
                # Never raise between collectives; a leftover dir is harmless.
                logger.warning("rotation could not remove %s: %s", path, exc)

    def _step_dirs(self) -> Iterator[Tuple[int, str]]:
        # Raw {run_id}-step{N} dirs, complete or not (rewind must see both).
        pat = re.compile(rf"^{re.escape(self.run_id)}-step(\d+)$")
        pattern = os.path.join(
            glob.escape(self.output_dir), f"{glob.escape(self.run_id)}-step*"
        )
        for path in glob.glob(pattern):
            m = pat.match(os.path.basename(path))
            if m and os.path.isdir(path):
                yield int(m.group(1)), path

    def _all_checkpoints(self) -> Iterator[Tuple[int, str]]:
        # Only dirs holding STATE_FILE count: a truncated save is never a target.
        for step, path in self._step_dirs():
            if os.path.isfile(self._state_path(path)):
                yield step, path

    def _max_step_on_disk(self) -> Optional[int]:
        steps = [step for step, _ in self._all_checkpoints()]
        return max(steps) if steps else None


__all__ = ["CheckpointManager"]
