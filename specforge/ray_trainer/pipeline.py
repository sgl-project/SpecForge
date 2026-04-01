"""
TrainingPipeline – orchestrates one global training step.

The pipeline decouples rollout (RolloutWorkerGroup) from training
(TrainWorkerGroup).  In colocated mode, rollout_group is None and
TrainWorkers run rollout internally.  In disaggregated mode,
rollout_group generates the RolloutBatch first, then train_group
trains on it.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Controls the per-step and per-epoch execution flow.

    Parameters
    ----------
    rollout_group : RolloutWorkerGroup or None
        None in colocated mode (TrainWorkers handle rollout internally).
    train_group   : TrainWorkerGroup
        Manages the FSDP draft model.
    args          : argparse.Namespace
        Full training arguments (for log_interval, eval_interval, etc.).
    """

    def __init__(
        self,
        rollout_group,          # Optional[RolloutWorkerGroup]
        train_group,            # TrainWorkerGroup
        args,
    ) -> None:
        self.rollout_group = rollout_group
        self.train_group = train_group
        self.args = args

    # ─────────────────────────────────────────────────────────────────────
    # Training step
    # ─────────────────────────────────────────────────────────────────────

    def run_train_step(
        self,
        data_batch: Optional[dict],
        global_step: int,
        tracker,
    ) -> dict:
        """
        Execute one complete training step.

        Colocated mode (rollout_group is None)
        ----------------------------------------
        TrainWorkers fetch their own data and run rollout internally.
        ``data_batch`` is ignored (each worker has its own DataLoader).

        Disaggregated mode (rollout_group is not None)
        ------------------------------------------------
        1. RolloutWorkerGroup generates RolloutBatch from data_batch.
        2. TrainWorkerGroup trains on the returned RolloutBatch.

        Returns
        -------
        Metrics dict (loss, acc, lr) from rank-0 TrainWorker.
        """
        if self.rollout_group is not None and data_batch is not None:
            # Disaggregated: run rollout first, then train
            rollout_refs = self.rollout_group.generate_rollout_batch(data_batch)
        else:
            rollout_refs = []

        metrics = self.train_group.train_step(rollout_refs, global_step)

        if metrics and global_step % self.args.log_interval == 0:
            if tracker is not None:
                tracker.log(metrics, step=global_step)

        return metrics or {}

    # ─────────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────────

    def run_eval(self, global_step: int, tracker) -> dict:
        """
        Run a full evaluation pass.

        TrainWorkers iterate over their own eval DataLoader internally.
        RolloutBatch is generated per batch within the worker for
        colocated mode, or via rollout_group for disaggregated mode.

        Returns
        -------
        Aggregated eval metrics dict from rank-0.
        """
        metrics = self.train_group.eval_step()
        if metrics and tracker is not None:
            eval_metrics = {f"eval/{k}": v for k, v in metrics.items()}
            tracker.log(eval_metrics, step=global_step)
        return metrics or {}

    # ─────────────────────────────────────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, step: int) -> str:
        """Delegate checkpoint saving to TrainWorkerGroup."""
        ckpt_path = self.train_group.save_checkpoint(epoch, step)
        logger.info(f"Checkpoint saved to: {ckpt_path}")
        return ckpt_path
