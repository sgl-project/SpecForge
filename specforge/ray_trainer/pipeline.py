"""
TrainingPipeline – orchestrates one global training step.

The pipeline is the **single point** that knows about colocated vs.
disaggregated mode.  It decouples rollout (RolloutWorkerGroup) from
training (TrainWorkerGroup).

* Colocated mode (rollout_group is None):
    Each TrainWorker loads the target model locally and runs rollout
    internally.  The pipeline simply calls train_group.train_step()
    without providing rollout data.

* Disaggregated mode — Ray transfer (--transfer-backend ray):
    Uses Ray object store to pass RolloutBatch between actors.

* Disaggregated mode — NCCL transfer (--transfer-backend nccl):
    GPU→GPU direct NCCL send/recv, no CPU round-trip.
    RolloutWorker sends directly to TrainWorker's GPU.
"""

import logging
import time
from collections import deque

import ray

logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(
        self,
        rollout_group,
        train_group,
        args,
        driver_dataloader=None,
    ) -> None:
        self.rollout_group = rollout_group
        self.train_group = train_group
        self.args = args
        self._disaggregate = rollout_group is not None
        self._transfer_backend = getattr(args, "transfer_backend", "ray")

        # Driver-side data iteration (disaggregated mode)
        self._driver_dataloader = driver_dataloader
        self._driver_iter = None

        # Prefetch state (Ray mode)
        self._prefetch_queue: deque = deque()
        self._current_ref = None
        self._current_count: int = 0
        self._current_index: int = 0

        # NCCL mode state
        self._nccl_pending: deque = deque()  # (send_ref, src_global_rank, split_count)
        self._nccl_current_src: int = -1
        self._nccl_current_count: int = 0
        self._nccl_current_index: int = 0
        self._enable_perf: bool = getattr(args, "enable_perf", False)

        self._next_tp_group: int = 0
        self._num_tp_groups: int = (
            rollout_group.num_tp_groups if rollout_group is not None else 0
        )
        self._rollout_batch_size: int = getattr(args, "rollout_batch_size", 1)
        self._max_prefetch: int = self._num_tp_groups * 2

    # ─────────────────────────────────────────────────────────────────────
    # Driver-side data fetching
    # ─────────────────────────────────────────────────────────────────────

    def set_epoch(self, epoch: int) -> None:
        if self._driver_dataloader is not None:
            sampler = getattr(self._driver_dataloader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            self._driver_iter = iter(self._driver_dataloader)
        # Clear stale prefetch state from previous epoch
        self._prefetch_queue.clear()
        self._nccl_pending.clear()
        self._current_ref = None
        self._current_count = 0
        self._current_index = 0
        self._nccl_current_src = -1
        self._nccl_current_count = 0
        self._nccl_current_index = 0

    def _fetch_batch_local(self):
        if self._driver_iter is None:
            return None
        try:
            batch = next(self._driver_iter)
            return {k: v.cpu() if hasattr(v, "cpu") else v for k, v in batch.items()}
        except StopIteration:
            return None

    def _fetch_multi_local(self, n: int):
        batches = []
        for _ in range(n):
            b = self._fetch_batch_local()
            if b is None:
                break
            batches.append(b)
        if not batches:
            return None, 0
        from specforge.ray_workers.worker_utils import pad_and_concat_batches

        return pad_and_concat_batches(batches)

    # ─────────────────────────────────────────────────────────────────────
    # Prefetch — Ray mode
    # ─────────────────────────────────────────────────────────────────────

    def prefetch_first_batch(self) -> None:
        if not self._disaggregate:
            return
        if self._transfer_backend == "nccl":
            self._fill_nccl_queue()
        else:
            self._fill_prefetch_queue()

    def _fill_prefetch_queue(self) -> None:
        while len(self._prefetch_queue) < self._max_prefetch:
            data_batch, actual_count = self._fetch_multi_local(self._rollout_batch_size)
            if data_batch is None:
                break
            ref = self.rollout_group.generate_rollout_batch_single(
                self._next_tp_group, data_batch
            )
            self._next_tp_group = (self._next_tp_group + 1) % self._num_tp_groups
            self._prefetch_queue.append((ref, actual_count))

    def _ensure_current_ref(self) -> bool:
        if self._current_ref is not None and self._current_index < self._current_count:
            return True
        if not self._prefetch_queue:
            return False
        self._current_ref, self._current_count = self._prefetch_queue.popleft()
        self._current_index = 0
        return True

    # ─────────────────────────────────────────────────────────────────────
    # Prefetch — NCCL mode
    # ─────────────────────────────────────────────────────────────────────

    def _fill_nccl_queue(self) -> None:
        """Dispatch rollout+send work until the NCCL queue is full.

        Each queue entry represents one logical training step:
        - dp_size separate rollout calls, each processing rollout_batch_size samples
        - Each rollout sends to one SP leader (one per DP group)
        - Pipeline round-robins across TP groups for each rollout call

        RolloutWorker is decoupled from DP — it always processes
        exactly rollout_batch_size samples and sends to a single dst.
        """
        dp_size = self.train_group.get_dp_size()
        sp_leader_ranks = self.train_group.get_sp_leader_ranks()

        while len(self._nccl_pending) < self._max_prefetch:
            # All dp_size rollout calls in one logical step use the same
            # TP group, because train_step_nccl_async sends a single
            # nccl_src_rank to all TrainWorkers.  Round-robin happens
            # across logical steps.
            tp_idx = self._next_tp_group
            src_global_rank = self.rollout_group.get_global_rank(tp_idx)
            self._next_tp_group = (self._next_tp_group + 1) % self._num_tp_groups

            # Dispatch dp_size rollout calls (one per DP group)
            send_refs = []
            per_dp_count = 0

            for dp_idx in range(dp_size):
                data_batch, actual_count = self._fetch_multi_local(
                    self._rollout_batch_size
                )
                if data_batch is None:
                    break
                per_dp_count = actual_count

                send_ref = self.rollout_group.generate_and_send_single(
                    tp_idx, data_batch, [sp_leader_ranks[dp_idx]]
                )
                send_refs.append(send_ref)

            if not send_refs:
                break

            self._nccl_pending.append((send_refs, src_global_rank, per_dp_count))

            if len(send_refs) < dp_size:
                break  # epoch boundary

    def _ensure_nccl_current(self) -> bool:
        """Ensure we have a current NCCL source to receive from."""
        if (
            self._nccl_current_src >= 0
            and self._nccl_current_index < self._nccl_current_count
        ):
            return True
        if not self._nccl_pending:
            return False
        _send_refs, self._nccl_current_src, self._nccl_current_count = (
            self._nccl_pending.popleft()
        )
        self._nccl_current_index = 0
        return True

    # ─────────────────────────────────────────────────────────────────────
    # Training step
    # ─────────────────────────────────────────────────────────────────────

    def run_train_step(self, global_step: int, tracker) -> dict:
        if not self._disaggregate:
            metrics = self.train_group.train_step(global_step=global_step)
            if metrics and global_step % self.args.log_interval == 0:
                if tracker is not None:
                    tracker.log(metrics, step=global_step)
            return metrics or {}

        if self._transfer_backend == "nccl":
            return self._run_train_step_nccl(global_step, tracker)
        else:
            return self._run_train_step_ray(global_step, tracker)

    def _run_train_step_ray(self, global_step: int, tracker) -> dict:
        """Disaggregated training step — Ray object store transfer."""
        if self._enable_perf:
            t0 = time.perf_counter()

        rollout_ref = None
        split_index = 0
        split_count = 1
        from_queue = False

        if self._ensure_current_ref():
            rollout_ref = self._current_ref
            split_index = self._current_index
            split_count = self._current_count
            self._current_index += 1
            from_queue = True
        else:
            data_batch = self._fetch_batch_local()
            if data_batch is not None:
                rollout_ref = self.rollout_group.generate_rollout_batch_single(
                    self._next_tp_group, data_batch
                )
                self._next_tp_group = (self._next_tp_group + 1) % self._num_tp_groups
        if self._enable_perf:
            t1 = time.perf_counter()

        self._fill_prefetch_queue()
        if self._enable_perf:
            t2 = time.perf_counter()

        train_refs = self.train_group.train_step_async(
            global_step=global_step,
            rollout_batch_ref=rollout_ref,
            split_index=split_index,
            split_count=split_count,
        )
        if self._enable_perf:
            t3 = time.perf_counter()

        metrics = ray.get(train_refs[0])

        if self._enable_perf:
            t4 = time.perf_counter()
            print(
                f"[PERF step {global_step}] "
                f"get_ref={'queue' if from_queue else 'sync'}={t1-t0:.3f}s  "
                f"fill_pf={t2-t1:.3f}s  "
                f"dispatch={t3-t2:.3f}s  "
                f"wait_train={t4-t3:.3f}s  "
                f"total={t4-t0:.3f}s  "
                f"pf_q={len(self._prefetch_queue)}  "
                f"split={split_index}/{split_count}",
                flush=True,
            )

        if metrics and global_step % self.args.log_interval == 0:
            if tracker is not None:
                tracker.log(metrics, step=global_step)
        return metrics or {}

    def _run_train_step_nccl(self, global_step: int, tracker) -> dict:
        """Disaggregated training step — NCCL GPU→GPU direct transfer."""
        if self._enable_perf:
            t0 = time.perf_counter()

        # 1. Ensure we have a NCCL source ready.
        nccl_src = -1
        split_index = 0
        split_count = 1
        is_first_split = False

        if self._ensure_nccl_current():
            nccl_src = self._nccl_current_src
            split_index = self._nccl_current_index
            split_count = self._nccl_current_count
            is_first_split = split_index == 0
            self._nccl_current_index += 1
        if self._enable_perf:
            t1 = time.perf_counter()

        # 2. Refill NCCL queue.
        self._fill_nccl_queue()
        if self._enable_perf:
            t2 = time.perf_counter()

        # 3. Dispatch training.
        #    Only the first split of each group does NCCL recv.
        #    Subsequent splits reuse cached data in TrainWorker.
        if is_first_split:
            train_refs = self.train_group.train_step_nccl_async(
                global_step=global_step,
                nccl_src_rank=nccl_src,
                split_index=split_index,
                split_count=split_count,
            )
        else:
            train_refs = self.train_group.train_step_nccl_async(
                global_step=global_step,
                nccl_src_rank=-1,
                split_index=split_index,
                split_count=split_count,
            )
        if self._enable_perf:
            t3 = time.perf_counter()

        # 4. Wait for training.
        metrics = ray.get(train_refs[0])

        if self._enable_perf:
            t4 = time.perf_counter()
            print(
                f"[PERF step {global_step}] "
                f"nccl_src={nccl_src}  "
                f"fill_q={t2-t1:.3f}s  "
                f"dispatch={t3-t2:.3f}s  "
                f"wait_train={t4-t3:.3f}s  "
                f"total={t4-t0:.3f}s  "
                f"nccl_q={len(self._nccl_pending)}  "
                f"split={split_index}/{split_count}  "
                f"recv={'yes' if is_first_split else 'cached'}",
                flush=True,
            )

        if metrics and global_step % self.args.log_interval == 0:
            if tracker is not None:
                tracker.log(metrics, step=global_step)
        return metrics or {}

    # ─────────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────────

    def run_eval(self, global_step: int, tracker) -> dict:
        metrics = self.train_group.eval_step()
        if metrics and tracker is not None:
            eval_metrics = {f"eval/{k}": v for k, v in metrics.items()}
            tracker.log(eval_metrics, step=global_step)
        return metrics or {}

    # ─────────────────────────────────────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, step: int) -> str:
        ckpt_path = self.train_group.save_checkpoint(epoch, step)
        logger.info(f"Checkpoint saved to: {ckpt_path}")
        return ckpt_path
