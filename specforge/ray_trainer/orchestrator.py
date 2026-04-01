"""
Eagle3RayOrchestrator – top-level training coordinator.

Runs in the driver process (not a Ray Actor).  It:
  1. Initialises the Ray cluster.
  2. Creates RolloutWorkerGroup and TrainWorkerGroup via resource_manager.
  3. Drives the training loop (epoch → step → eval → checkpoint).
  4. Manages the experiment Tracker and graceful shutdown.

This class is the Ray-based counterpart of the ``main()`` function in
``scripts/train_eagle3.py``.
"""

import logging
import os

import ray
from accelerate.utils import set_seed

logger = logging.getLogger(__name__)


class Eagle3RayOrchestrator:
    """
    Top-level orchestrator for Ray-based Eagle3 training.

    Unlike the torchrun-based script, the orchestrator does not own any
    model parameters.  It coordinates control flow only, delegating all
    GPU compute to TrainWorker and RolloutWorker Actors.
    """

    def __init__(self, args) -> None:
        self.args = args

        # ── 1. Init Ray ────────────────────────────────────────────────────
        ray_address = getattr(args, "ray_address", None)
        ray_namespace = getattr(args, "ray_namespace", "specforge")
        ray_num_gpus = getattr(args, "ray_num_gpus", None)

        ray_init_kwargs = dict(
            address=ray_address,
            namespace=ray_namespace,
            ignore_reinit_error=True,
            logging_level=logging.WARNING,
        )
        if ray_address is None and ray_num_gpus is not None:
            ray_init_kwargs["num_gpus"] = ray_num_gpus

        ray.init(**ray_init_kwargs)
        print("Ray initialised.")

        # ── 2. Build draft model config (for aux_hidden_states_layers) ─────
        from specforge.args import SGLangBackendArgs
        from specforge.modeling import AutoDraftModelConfig
        from specforge.utils import create_draft_config_from_target

        if args.draft_model_config is None:
            auto_config_path = create_draft_config_from_target(
                target_model_path=args.target_model_path,
                cache_dir=getattr(args, "model_download_dir", None),
            )
            draft_model_config = AutoDraftModelConfig.from_file(auto_config_path)
        else:
            draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)

        # Resolve aux_hidden_states_layers from draft model config
        aux_hidden_states_layers = None
        if (
            hasattr(draft_model_config, "eagle_config")
            and draft_model_config.eagle_config is not None
            and "eagle_aux_hidden_state_layer_ids" in draft_model_config.eagle_config
        ):
            aux_hidden_states_layers = draft_model_config.eagle_config[
                "eagle_aux_hidden_state_layer_ids"
            ]

        # ── 3. Build SGLang backend kwargs ─────────────────────────────────
        if getattr(args, "target_model_backend", "sglang") == "sglang":
            sglang_backend_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()
        else:
            sglang_backend_kwargs = {}

        # ── 4. Create worker groups ────────────────────────────────────────
        from specforge.ray_trainer.resource_manager import build_worker_groups
        print('##')
        self.rollout_group, self.train_group = build_worker_groups(
            args=args,
            sglang_backend_kwargs=sglang_backend_kwargs,
            aux_hidden_states_layers=aux_hidden_states_layers,
        )
        print('######')
        # ── 5. Build pipeline ──────────────────────────────────────────────
        from specforge.ray_trainer.pipeline import TrainingPipeline
        import pdb
        pdb.set_trace()
        self.pipeline = TrainingPipeline(
            rollout_group=self.rollout_group,
            train_group=self.train_group,
            args=args,
        )

        # ── 6. Retrieve dataset info and compute total_steps ───────────────
        dataset_info = self.train_group.get_dataset_info()
        self.start_epoch: int = dataset_info["start_epoch"]
        self.global_step: int = dataset_info["global_step"]
        self.steps_per_epoch: int = dataset_info["train_steps_per_epoch"]

        if args.total_steps is None:
            args.total_steps = self._compute_total_steps()
            logger.info(f"Computed total_steps = {args.total_steps}")

        # ── 7. Init tracker ────────────────────────────────────────────────
        from specforge.tracker import create_tracker, get_tracker_class

        tracker_class = get_tracker_class(getattr(args, "report_to", "none"))
        self.tracker = create_tracker(args, args.output_dir) if tracker_class else None

        print(
            f"Orchestrator ready. start_epoch={self.start_epoch}, "
            f"global_step={self.global_step}, "
            f"steps_per_epoch={self.steps_per_epoch}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Main training loop
    # ─────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Drive the full training loop."""
        args = self.args
        num_epochs = args.num_epochs
        max_num_steps = getattr(args, "max_num_steps", None)
        eval_interval = getattr(args, "eval_interval", 5000)
        save_interval = getattr(args, "save_interval", 5000)

        print('####:',args)
        global_step = self.global_step
        done = False
        for epoch in range(self.start_epoch, num_epochs):
            self.train_group.set_epoch(epoch)
            steps_to_skip = global_step % self.steps_per_epoch if epoch == self.start_epoch else 0

            for step_in_epoch in range(self.steps_per_epoch):
                if step_in_epoch < steps_to_skip:
                    # Skip already-completed steps for resume
                    continue

                global_step += 1

                metrics = self.pipeline.run_train_step(
                    data_batch=None,   # each TrainWorker has its own DataLoader
                    global_step=global_step,
                    tracker=self.tracker,
                )

                if global_step % args.log_interval == 0:
                    loss_str = "  ".join(
                        f"{k}={v:.4f}" for k, v in metrics.items()
                        if "ploss" in k or "lr" in k
                    )
                    logger.info(f"[step {global_step}] {loss_str}")

                if global_step % eval_interval == 0:
                    self.pipeline.run_eval(global_step, self.tracker)

                if global_step % save_interval == 0:
                    self.pipeline.save_checkpoint(epoch, global_step)

                if max_num_steps is not None and global_step >= max_num_steps:
                    done = True
                    break

            if done:
                break

        # Final checkpoint
        logger.info("Training complete. Saving final checkpoint …")
        self.pipeline.save_checkpoint(epoch, global_step)

        if self.tracker is not None:
            self.tracker.finish()

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _compute_total_steps(self) -> int:
        """Compute total training steps from dataset size and num_epochs."""
        args = self.args
        remaining_epochs = args.num_epochs - self.start_epoch
        return remaining_epochs * self.steps_per_epoch

    def shutdown(self) -> None:
        """Gracefully shut down workers and the Ray cluster."""
        logger.info("Shutting down worker groups …")
        try:
            self.train_group.shutdown()
        except Exception as e:
            logger.warning(f"TrainWorkerGroup shutdown error: {e}")

        if self.rollout_group is not None:
            try:
                self.rollout_group.shutdown()
            except Exception as e:
                logger.warning(f"RolloutWorkerGroup shutdown error: {e}")

        ray.shutdown()
        logger.info("Ray shutdown complete.")
