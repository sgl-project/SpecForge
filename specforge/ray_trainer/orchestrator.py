"""
RayOrchestrator – top-level training coordinator.

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

logger = logging.getLogger(__name__)


class RayOrchestrator:
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
        logger.info("Ray initialised.")

        # ── 2. Resolve draft model config and layer capture IDs ─────────
        from specforge.args import SGLangBackendArgs

        method = getattr(args, "method", "eagle3")
        capture_layer_ids = None

        if method == "eagle3":
            from specforge.modeling import AutoDraftModelConfig
            from specforge.utils import create_draft_config_from_target

            if args.draft_model_config is None:
                auto_config_path = create_draft_config_from_target(
                    target_model_path=args.target_model_path,
                    cache_dir=getattr(args, "model_download_dir", None),
                )
                draft_model_config = AutoDraftModelConfig.from_file(auto_config_path)
            else:
                draft_model_config = AutoDraftModelConfig.from_file(
                    args.draft_model_config
                )

            if (
                hasattr(draft_model_config, "eagle_config")
                and draft_model_config.eagle_config is not None
                and "eagle_aux_hidden_state_layer_ids"
                in draft_model_config.eagle_config
            ):
                capture_layer_ids = draft_model_config.eagle_config[
                    "eagle_aux_hidden_state_layer_ids"
                ]

        elif method == "dflash":
            if args.draft_model_config is not None:
                from transformers import AutoConfig

                draft_model_config = AutoConfig.from_pretrained(args.draft_model_config)
                dflash_cfg = getattr(draft_model_config, "dflash_config", None) or {}
                capture_layer_ids = dflash_cfg.get("target_layer_ids", None)
            else:
                # Auto-compute from target model
                from transformers import AutoConfig

                target_config = AutoConfig.from_pretrained(args.target_model_path)
                num_target_layers = target_config.num_hidden_layers
                num_draft_layers = getattr(args, "num_draft_layers", 5)
                from specforge.modeling.draft.dflash import build_target_layer_ids

                capture_layer_ids = build_target_layer_ids(
                    num_target_layers, num_draft_layers
                )

        # ── 3. Build SGLang backend kwargs ─────────────────────────────────
        if getattr(args, "target_model_backend", "sglang") == "sglang":
            sglang_backend_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()
        else:
            sglang_backend_kwargs = {}

        # ── 4. Pre-build dataset in driver (before GPU actors) ────────────
        # HuggingFace datasets.map(num_proc=N) uses fork-based multiprocessing.
        # If we let each TrainWorker build the dataset after CUDA init, the
        # fork will deadlock.  Even building before CUDA init in each actor
        # causes resource contention (N actors × M num_proc = too many forks).
        # Solution: build once here in the driver, write to cache.  Workers
        # will hit the cache and skip the expensive map() call.
        train_dataset_len, train_dataset = self._prebuild_dataset_cache(args)

        # ── 4b. Pre-compute total_steps (needed by TrainWorker's LR scheduler)
        if args.total_steps is None and train_dataset_len > 0:
            train_num_gpus = getattr(args, "train_num_gpus", None)
            if train_num_gpus is None:
                import torch

                train_num_gpus = torch.cuda.device_count()
            sp_ulysses = getattr(args, "train_sp_ulysses_size", 1)
            sp_ring = getattr(args, "train_sp_ring_size", 1)
            sp_size = sp_ulysses * sp_ring
            target_batch_size = args.batch_size
            dp_size = train_num_gpus // max(sp_size, 1)
            steps_per_epoch = len(
                range(0, train_dataset_len, target_batch_size * dp_size)
            )
            args.total_steps = args.num_epochs * steps_per_epoch
            logger.info(
                f"Pre-computed total_steps = {args.total_steps} "
                f"(dataset_len={train_dataset_len}, steps_per_epoch={steps_per_epoch})"
            )

        # ── 4c. Build driver-side DataLoader for disaggregated mode ───────
        #    In disaggregated mode the driver feeds data directly to
        #    RolloutWorkers, bypassing TrainWorker actor fetch calls.
        self._driver_dataloader = None
        if getattr(args, "disaggregate", False):
            if train_dataset is None:
                raise ValueError(
                    "Disaggregated mode requires a training dataset. "
                    "Ensure --train-data-path is set and dataset building succeeded."
                )
            from torch.utils.data import DataLoader

            from specforge.data.utils import DataCollatorWithPadding

            self._driver_dataloader = DataLoader(
                train_dataset,
                batch_size=target_batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                prefetch_factor=2,
                collate_fn=DataCollatorWithPadding(sp_degree=1, ulysses_degree=1),
                drop_last=True,
            )

        # ── 5. Create worker groups ────────────────────────────────────────
        from specforge.ray_trainer.resource_manager import build_worker_groups

        self.rollout_group, self.train_group = build_worker_groups(
            args=args,
            sglang_backend_kwargs=sglang_backend_kwargs,
            capture_layer_ids=capture_layer_ids,
        )
        # ── 5. Build pipeline ──────────────────────────────────────────────
        from specforge.ray_trainer.pipeline import TrainingPipeline

        self.pipeline = TrainingPipeline(
            rollout_group=self.rollout_group,
            train_group=self.train_group,
            args=args,
            driver_dataloader=self._driver_dataloader,
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

        logger.info(
            f"Orchestrator ready. start_epoch={self.start_epoch}, "
            f"global_step={self.global_step}, "
            f"steps_per_epoch={self.steps_per_epoch}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Main training loop
    # ─────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Drive the full training loop."""
        from tqdm import tqdm

        args = self.args
        num_epochs = args.num_epochs
        max_num_steps = getattr(args, "max_num_steps", None)
        eval_interval = getattr(args, "eval_interval", 5000)
        save_interval = getattr(args, "save_interval", 5000)

        global_step = self.global_step
        total_steps = args.total_steps or (num_epochs * self.steps_per_epoch)
        pbar = tqdm(
            total=total_steps,
            initial=global_step,
            desc="Training",
            dynamic_ncols=True,
        )
        done = False
        for epoch in range(self.start_epoch, num_epochs):
            self.train_group.set_epoch(epoch)
            self.pipeline.set_epoch(epoch)
            # Prime the async pipeline on first epoch (after set_epoch
            # initializes the driver DataLoader iterator).
            if epoch == self.start_epoch:
                self.pipeline.prefetch_first_batch()
            steps_to_skip = (
                global_step % self.steps_per_epoch if epoch == self.start_epoch else 0
            )

            for step_in_epoch in range(self.steps_per_epoch):
                if step_in_epoch < steps_to_skip:
                    # Skip already-completed steps for resume
                    continue

                global_step += 1

                metrics = self.pipeline.run_train_step(
                    global_step=global_step,
                    tracker=self.tracker,
                )

                pbar.update(1)
                if metrics:
                    accs = [v for k, v in metrics.items() if "/acc_" in k]
                    losses = [v for k, v in metrics.items() if "/ploss_" in k]
                    pbar_metrics = {}
                    if losses:
                        pbar_metrics["loss"] = f"{sum(losses)/len(losses):.4f}"
                    if accs:
                        pbar_metrics["acc"] = f"{sum(accs)/len(accs):.4f}"
                    pbar.set_postfix(pbar_metrics)

                if global_step % args.log_interval == 0:
                    loss_str = "  ".join(
                        f"{k}={v:.4f}"
                        for k, v in metrics.items()
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

        pbar.close()

        # Final checkpoint
        logger.info("Training complete. Saving final checkpoint …")
        self.pipeline.save_checkpoint(epoch, global_step)

        if self.tracker is not None:
            self.tracker.finish()

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _prebuild_dataset_cache(self, args) -> tuple:
        """
        Build train/eval datasets and vocab mapping in the driver process.

        This populates the HuggingFace datasets cache so that TrainWorker
        actors can load the processed dataset from cache without calling
        dataset.map(num_proc=N), which would fork after CUDA init and
        deadlock.
        """
        import hashlib

        from transformers import AutoTokenizer

        from datasets import Dataset
        from specforge.data import build_eagle3_dataset, generate_vocab_mapping_file
        from specforge.modeling import AutoDraftModelConfig
        from specforge.utils import safe_conversations_generator

        train_data_path = getattr(args, "train_data_path", None)
        if train_data_path is None:
            return 0, None

        target_model_path = args.target_model_path
        chat_template = args.chat_template
        max_length = args.max_length
        cache_dir = args.cache_dir
        build_dataset_num_proc = args.build_dataset_num_proc
        is_vlm = getattr(args, "is_vlm", False)
        is_preformatted = getattr(args, "is_preformatted", False)
        train_only_last_turn = getattr(args, "train_only_last_turn", False)
        trust_remote_code = getattr(args, "trust_remote_code", False)
        eval_data_path = getattr(args, "eval_data_path", None)

        cache_params_string = (
            f"{train_data_path}-{max_length}-{chat_template}-{target_model_path}"
        )
        cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

        logger.info("Pre-building dataset cache in driver process ...")

        tokenizer = AutoTokenizer.from_pretrained(
            target_model_path, trust_remote_code=trust_remote_code
        )

        train_dataset_raw = Dataset.from_generator(
            generator=safe_conversations_generator,
            gen_kwargs={"file_path": train_data_path},
        )

        train_eagle3_dataset = build_eagle3_dataset(
            dataset=train_dataset_raw,
            tokenizer=tokenizer,
            chat_template=chat_template,
            max_length=max_length,
            cache_dir=os.path.join(cache_dir, "processed_dataset"),
            cache_key=cache_key,
            is_vlm=is_vlm,
            is_preformatted=is_preformatted,
            num_proc=build_dataset_num_proc,
            train_only_last_turn=train_only_last_turn,
        )

        # Build vocab mapping (Eagle3 only — DFlash uses full target vocab)
        method = getattr(args, "method", "eagle3")
        if method == "eagle3":
            draft_model_config_path = getattr(args, "draft_model_config", None)
            if draft_model_config_path is not None:
                dmc = AutoDraftModelConfig.from_file(draft_model_config_path)
            else:
                from specforge.utils import create_draft_config_from_target

                auto_cfg_path = create_draft_config_from_target(
                    target_model_path=target_model_path,
                    cache_dir=getattr(args, "model_download_dir", None),
                )
                dmc = AutoDraftModelConfig.from_file(auto_cfg_path)

            generate_vocab_mapping_file(
                dataset=train_eagle3_dataset,
                target_vocab_size=dmc.vocab_size,
                draft_vocab_size=dmc.draft_vocab_size,
                cache_dir=os.path.join(cache_dir, "vocab_mapping"),
                cache_key=cache_key,
            )

        # Eval dataset
        if eval_data_path is not None:
            eval_dataset_raw = Dataset.from_generator(
                generator=safe_conversations_generator,
                gen_kwargs={"file_path": eval_data_path},
            )
            build_eagle3_dataset(
                eval_dataset_raw,
                tokenizer,
                chat_template,
                max_length,
                is_vlm=is_vlm,
                num_proc=build_dataset_num_proc,
                is_preformatted=is_preformatted,
                train_only_last_turn=train_only_last_turn,
            )

        logger.info("Dataset cache ready.")
        return len(train_eagle3_dataset), train_eagle3_dataset

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
        except Exception:
            logger.exception("TrainWorkerGroup shutdown error")

        if self.rollout_group is not None:
            try:
                self.rollout_group.shutdown()
            except Exception:
                logger.exception("RolloutWorkerGroup shutdown error")

        ray.shutdown()
        logger.info("Ray shutdown complete.")
