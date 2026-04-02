"""
Resource manager for Ray-based Eagle3 training.

Manages placement groups and worker actor lifecycle for both
RolloutWorkerGroup (target model inference) and TrainWorkerGroup
(draft model training).

Architecture
------------
* **Colocated mode** (disaggregate=False):
    Only TrainWorkerGroup is created.  Each TrainWorker loads the target
    model locally and runs rollout internally.  No RolloutWorkerGroup.

* **Disaggregated mode** (disaggregate=True):
    Both groups are created on separate GPU pools.  The pipeline drives
    the data flow: RolloutWorkerGroup produces RolloutBatch ObjectRefs,
    which are passed to TrainWorkerGroup.train_step().
"""

import logging
import socket
from typing import List, Optional, Tuple

import ray

logger = logging.getLogger(__name__)


def _get_head_node_ip() -> str:
    """Return the IP address of the current (driver/head) node."""
    return socket.gethostbyname(socket.gethostname())


# ─────────────────────────────────────────────────────────────────────────
# RolloutWorkerGroup
# ─────────────────────────────────────────────────────────────────────────


class RolloutWorkerGroup:
    """
    Manages a pool of RolloutWorker Ray Actors (disaggregated mode only).

    All workers within the same TP group share a rendezvous port and form
    a single torch.distributed process group on the rollout side.  Only
    the tp_rank=0 worker in each TP group returns a populated RolloutBatch;
    the others participate in collective ops but return None.
    """

    def __init__(
        self,
        num_workers: int,
        tp_size: int,
        target_model_path: str,
        backend: str,
        sglang_backend_kwargs: dict,
        capture_layer_ids=None,
        method: str = "eagle3",
        is_vlm: bool = False,
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        model_download_dir: Optional[str] = None,
        timeout_minutes: int = 20,
        placement_group=None,
        placement_group_bundle_offset: int = 0,
        # NCCL transfer mode params
        transfer_backend: str = "ray",
        global_rank_offset: int = 0,
        global_world_size: int = 0,
        global_master_addr: str = "",
        global_master_port: int = 0,
        rollout_ranks: Optional[list] = None,
    ) -> None:
        from specforge.distributed_ray import get_free_port
        from specforge.ray_workers import RolloutWorker

        assert (
            num_workers % tp_size == 0
        ), f"num_workers ({num_workers}) must be divisible by tp_size ({tp_size})"
        self.num_workers = num_workers
        self.tp_size = tp_size
        self.num_tp_groups = num_workers // tp_size
        self._transfer_backend = transfer_backend
        self._global_rank_offset = global_rank_offset
        master_addr = _get_head_node_ip()

        # One rendezvous port per TP group (used in Ray mode)
        master_ports = [get_free_port() for _ in range(self.num_tp_groups)]

        self._workers: List = []
        for rank in range(num_workers):
            tp_group_idx = rank // tp_size
            tp_rank = rank % tp_size

            kwargs = dict(
                rank=tp_rank,
                world_size=tp_size,
                tp_size=tp_size,
                tp_rank=tp_rank,
                master_addr=master_addr,
                master_port=master_ports[tp_group_idx],
                target_model_path=target_model_path,
                backend=backend,
                sglang_backend_kwargs=sglang_backend_kwargs,
                capture_layer_ids=capture_layer_ids,
                method=method,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                model_download_dir=model_download_dir,
                is_vlm=is_vlm,
                timeout_minutes=timeout_minutes,
                transfer_backend=transfer_backend,
            )

            if transfer_backend == "nccl":
                kwargs.update(
                    global_rank=global_rank_offset + rank,
                    global_world_size=global_world_size,
                    global_master_addr=global_master_addr,
                    global_master_port=global_master_port,
                    rollout_ranks=rollout_ranks,
                )

            if placement_group is not None:
                bundle_index = placement_group_bundle_offset + rank
                actor = RolloutWorker.options(
                    placement_group=placement_group,
                    placement_group_bundle_index=bundle_index,
                ).remote(**kwargs)
            else:
                actor = RolloutWorker.remote(**kwargs)
            self._workers.append(actor)

        if transfer_backend != "nccl":
            # In NCCL mode, defer ready-check until all actors are created
            # (global init_process_group needs all ranks to join).
            ray.get([w.is_ready.remote() for w in self._workers])
        logger.info(f"RolloutWorkerGroup: {num_workers} workers created.")

    def wait_ready(self) -> None:
        """Block until all workers are ready (used for deferred init)."""
        ray.get([w.is_ready.remote() for w in self._workers])

    def generate_rollout_batch(self, data_batch: dict):
        """
        Distribute *data_batch* to all TP groups and invoke rollout in parallel.

        Each TP group processes the full ``data_batch`` (the target model is
        TP-sharded across the group).  Only the tp_rank=0 worker in each
        group returns a populated RolloutBatch; other workers return None.

        Returns
        -------
        List of Ray ObjectRefs — one per TP group.
        """
        refs = []
        for tp_group_idx in range(self.num_tp_groups):
            group_refs = []
            for tp_rank in range(self.tp_size):
                rank = tp_group_idx * self.tp_size + tp_rank
                ref = self._workers[rank].generate_rollout_batch.remote(
                    data_batch["input_ids"],
                    data_batch["attention_mask"],
                    data_batch["loss_mask"],
                    data_batch.get("pixel_values"),
                    data_batch.get("image_grid_thw"),
                )
                group_refs.append(ref)
            # Only tp_rank=0 returns non-None; pick the first ref
            refs.append(group_refs[0])
        return refs

    def generate_rollout_batch_single(self, tp_group_idx: int, data_batch: dict):
        """Send batch to one specific TP group and return its ObjectRef."""
        group_refs = []
        for tp_rank in range(self.tp_size):
            rank = tp_group_idx * self.tp_size + tp_rank
            ref = self._workers[rank].generate_rollout_batch.remote(
                data_batch["input_ids"],
                data_batch["attention_mask"],
                data_batch["loss_mask"],
                data_batch.get("pixel_values"),
                data_batch.get("image_grid_thw"),
            )
            group_refs.append(ref)
        return group_refs[0]  # tp_rank=0 returns non-None

    def generate_and_send_single(
        self, tp_group_idx: int, data_batch: dict, dst_ranks: list
    ):
        """Run rollout on one TP group and NCCL-send result to dst_ranks.

        Returns an ObjectRef for the completion signal (no data payload).
        """
        group_refs = []
        for tp_rank in range(self.tp_size):
            rank = tp_group_idx * self.tp_size + tp_rank
            ref = self._workers[rank].generate_and_send.remote(
                data_batch["input_ids"],
                data_batch["attention_mask"],
                data_batch["loss_mask"],
                dst_ranks,
                data_batch.get("pixel_values"),
                data_batch.get("image_grid_thw"),
            )
            group_refs.append(ref)
        return group_refs[0]

    def get_global_rank(self, tp_group_idx: int) -> int:
        """Return the global rank of tp_rank=0 in the given TP group."""
        return self._global_rank_offset + tp_group_idx * self.tp_size

    def shutdown(self) -> None:
        """Shutdown all workers."""
        ray.get([w.shutdown.remote() for w in self._workers])
        logger.info("RolloutWorkerGroup shutdown complete.")


# ─────────────────────────────────────────────────────────────────────────
# TrainWorkerGroup
# ─────────────────────────────────────────────────────────────────────────


class TrainWorkerGroup:
    """
    Manages a pool of TrainWorker Ray Actors supporting DP + SP + TP.

    GPU allocation
    --------------
    num_workers = world_size = dp_size * tp_size

    SP does NOT add GPUs; it re-partitions existing ranks along the
    sequence dimension (handled inside torch.distributed via yunchang).

    Rank assignment (mirrors init_device_mesh in distributed.py):
        tp_rank  = rank % tp_size
        dp_rank  = rank // tp_size
        sp_size  = sp_ulysses_size * sp_ring_size
        sp_rank       = rank % sp_size
        draft_dp_rank = rank // sp_size
    """

    def __init__(
        self,
        num_workers: int,
        tp_size: int,
        sp_ulysses_size: int,
        sp_ring_size: int,
        draft_model_args: dict,
        train_hparams: dict,
        output_dir: str,
        ckpt_dir: Optional[str],
        placement_group=None,
        placement_group_bundle_offset: int = 0,
        # NCCL transfer mode params
        transfer_backend: str = "ray",
        global_rank_offset: int = 0,
        global_world_size: int = 0,
        global_master_addr: str = "",
        global_master_port: int = 0,
        train_ranks: Optional[list] = None,
    ) -> None:
        from specforge.distributed_ray import get_free_port
        from specforge.ray_workers import TrainWorker

        sp_size = sp_ulysses_size * sp_ring_size
        assert (
            num_workers % tp_size == 0
        ), f"num_workers ({num_workers}) must be divisible by tp_size ({tp_size})"
        assert (
            num_workers % sp_size == 0
        ), f"num_workers ({num_workers}) must be divisible by sp_size ({sp_size})"

        self.num_workers = num_workers
        self.tp_size = tp_size
        self._transfer_backend = transfer_backend
        self._global_rank_offset = global_rank_offset
        self._sp_size = sp_size
        self._dp_size = num_workers // max(sp_size, 1)
        master_addr = _get_head_node_ip()
        master_port = get_free_port()

        self._workers: List = []
        for rank in range(num_workers):
            kwargs = dict(
                rank=rank,
                world_size=num_workers,
                tp_size=tp_size,
                sp_ulysses_size=sp_ulysses_size,
                sp_ring_size=sp_ring_size,
                master_addr=master_addr,
                master_port=master_port,
                output_dir=output_dir,
                ckpt_dir=ckpt_dir,
                transfer_backend=transfer_backend,
                **draft_model_args,
                **train_hparams,
            )

            if transfer_backend == "nccl":
                kwargs.update(
                    global_rank=global_rank_offset + rank,
                    global_world_size=global_world_size,
                    global_master_addr=global_master_addr,
                    global_master_port=global_master_port,
                    train_ranks=train_ranks,
                )

            if placement_group is not None:
                bundle_index = placement_group_bundle_offset + rank
                actor = TrainWorker.options(
                    placement_group=placement_group,
                    placement_group_bundle_index=bundle_index,
                ).remote(**kwargs)
            else:
                actor = TrainWorker.remote(**kwargs)
            self._workers.append(actor)

        if transfer_backend != "nccl":
            ray.get([w.is_ready.remote() for w in self._workers])
        logger.info(f"TrainWorkerGroup: {num_workers} workers created.")

    def wait_ready(self) -> None:
        """Block until all workers are ready (used for deferred init)."""
        ray.get([w.is_ready.remote() for w in self._workers])

    # ── Dataset info & epoch management ────────────────────────────────

    def get_dataset_info(self) -> dict:
        """Return dataset info from rank-0 worker."""
        return ray.get(self._workers[0].get_dataset_info.remote())

    def set_epoch(self, epoch: int) -> None:
        """Set epoch on all workers (for DistributedSampler)."""
        ray.get([w.set_epoch.remote(epoch) for w in self._workers])

    # ── Data fetching (disaggregated mode) ───────────────────────────

    def fetch_batch(self) -> Optional[dict]:
        """
        Fetch the next data batch from rank-0's DataLoader.

        All workers advance their iterators (to keep DistributedSampler
        in sync), but only rank-0's batch is returned to the caller.
        """
        fetch_refs = [w.fetch_batch.remote() for w in self._workers]
        results = ray.get(fetch_refs)
        return results[0]  # rank-0's batch

    def fetch_batch_multi(self, n: int) -> Optional[dict]:
        """Fetch *n* batches, pad to the same seq_len, and concatenate along dim=0.

        Returns None if the iterator is exhausted before any batch is
        fetched.  If fewer than *n* batches are available (epoch end),
        returns whatever was accumulated.
        """
        from specforge.ray_workers.worker_utils import pad_and_concat_batches

        batches = []
        for _ in range(n):
            b = self.fetch_batch()
            if b is None:
                break
            batches.append(b)
        if not batches:
            return None
        if len(batches) == 1:
            return batches[0]
        merged, _ = pad_and_concat_batches(batches)
        return merged

    # ── Training ───────────────────────────────────────────────────────

    def train_step(
        self,
        global_step: int,
        rollout_batch_ref=None,
    ) -> dict:
        """
        Execute one training step across all workers.

        Args:
            global_step: Current global training step.
            rollout_batch_ref: (disaggregated mode) Ray ObjectRef pointing
                to a RolloutBatch.  All workers receive the same ref and
                internally shard by tp_rank.  None in colocated mode.

        Returns:
            Metrics dict from rank-0.
        """
        step_refs = [
            w.run_step.remote(global_step, rollout_batch_ref=rollout_batch_ref)
            for w in self._workers
        ]
        results = ray.get(step_refs)
        return results[0]

    def train_step_async(
        self,
        global_step: int,
        rollout_batch_ref=None,
        split_index: int = 0,
        split_count: int = 1,
    ) -> list:
        """Like train_step but returns ObjectRefs without blocking."""
        return [
            w.run_step.remote(
                global_step,
                rollout_batch_ref=rollout_batch_ref,
                split_index=split_index,
                split_count=split_count,
            )
            for w in self._workers
        ]

    def train_step_nccl_async(
        self,
        global_step: int,
        nccl_src_rank: int,
        split_index: int = 0,
        split_count: int = 1,
    ) -> list:
        """Start training with NCCL recv from a specific rollout worker.

        In DP>1 mode, only SP leaders recv from RolloutWorker.
        Non-leaders recv via SP broadcast (handled inside TrainWorker).
        All workers get the same nccl_src_rank — each worker internally
        checks its sp_rank to decide whether to recv or broadcast-recv.
        """
        return [
            w.run_step.remote(
                global_step,
                nccl_src_rank=nccl_src_rank,
                split_index=split_index,
                split_count=split_count,
            )
            for w in self._workers
        ]

    def get_global_ranks(self) -> list:
        """Return list of global ranks for all train workers."""
        return [self._global_rank_offset + i for i in range(self.num_workers)]

    def get_sp_leader_ranks(self) -> list:
        """Return global ranks of SP leaders (sp_rank=0 in each SP/DP group).

        With train_ranks laid out as [dp0_sp0, dp0_sp1, ..., dp1_sp0, ...],
        SP leaders are at indices 0, sp_size, 2*sp_size, ...
        """
        return [
            self._global_rank_offset + i * self._sp_size for i in range(self._dp_size)
        ]

    def get_dp_size(self) -> int:
        """Return the number of DP groups."""
        return self._dp_size

    # ── Evaluation ─────────────────────────────────────────────────────

    def eval_step(self, rollout_batch_ref=None) -> dict:
        """
        Run eval across all workers and return rank-0 metrics.

        In colocated mode (rollout_batch_ref=None), each worker iterates
        over its full eval DataLoader internally.

        In disaggregated mode, receives a single RolloutBatch ObjectRef.
        """
        eval_refs = [
            w.run_eval_step.remote(rollout_batch_ref=rollout_batch_ref)
            for w in self._workers
        ]
        results = ray.get(eval_refs)
        return results[0]

    # ── Checkpoint ─────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, step: int) -> str:
        """Save checkpoint; returns path from rank-0."""
        ckpt_refs = [w.save_checkpoint.remote(epoch, step) for w in self._workers]
        results = ray.get(ckpt_refs)
        return results[0]

    # ── Lifecycle ──────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Shutdown all workers."""
        ray.get([w.shutdown.remote() for w in self._workers])
        logger.info("TrainWorkerGroup shutdown complete.")


# ─────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────


def _build_common_args(args, capture_layer_ids):
    """Build the common kwargs dicts shared by colocated and disaggregated modes."""
    draft_model_args = dict(
        target_model_path=args.target_model_path,
        draft_model_config_path=getattr(args, "draft_model_config", None),
        attention_backend=args.attention_backend,
        embedding_key=args.embedding_key,
        train_data_path=args.train_data_path,
        eval_data_path=getattr(args, "eval_data_path", None),
        chat_template=args.chat_template,
        max_length=args.max_length,
        batch_size=args.batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        build_dataset_num_proc=args.build_dataset_num_proc,
        is_preformatted=getattr(args, "is_preformatted", False),
        train_only_last_turn=getattr(args, "train_only_last_turn", False),
        is_vlm=getattr(args, "is_vlm", False),
        cache_dir=args.cache_dir,
        torch_dtype="bfloat16",
        trust_remote_code=getattr(args, "trust_remote_code", False),
        model_download_dir=getattr(args, "model_download_dir", None),
        timeout_minutes=getattr(args, "dist_timeout", 20),
        seed=getattr(args, "seed", 0),
        method=getattr(args, "method", "eagle3"),
        capture_layer_ids=capture_layer_ids,
        lm_head_key=getattr(args, "lm_head_key", "lm_head.weight"),
    )

    train_hparams = dict(
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=getattr(args, "total_steps", None) or 0,
        ttt_length=args.ttt_length,
        draft_accumulation_steps=args.draft_accumulation_steps,
        log_interval=getattr(args, "log_interval", 50),
        resume=getattr(args, "resume", False),
        block_size=getattr(args, "block_size", 16),
        num_anchors=getattr(args, "num_anchors", 512),
        loss_decay_gamma=getattr(args, "loss_decay_gamma", None),
        mask_token_id=getattr(args, "mask_token_id", None),
        num_draft_layers=getattr(args, "num_draft_layers", 5),
    )

    return draft_model_args, train_hparams


def _build_colocated_groups(
    args,
    sglang_backend_kwargs,
    draft_model_args,
    train_hparams,
    train_tp_size,
    train_sp_ulysses_size,
    train_sp_ring_size,
) -> Tuple[None, TrainWorkerGroup]:
    """Create TrainWorkerGroup for colocated mode (rollout + train on same GPU)."""
    train_num_gpus = getattr(args, "train_num_gpus", None)
    if train_num_gpus is None:
        import torch

        train_num_gpus = torch.cuda.device_count()

    train_hparams_coloc = dict(train_hparams)
    train_hparams_coloc["disaggregate"] = False
    train_hparams_coloc["target_model_backend"] = getattr(
        args, "target_model_backend", "sglang"
    )
    train_hparams_coloc["sglang_backend_kwargs"] = sglang_backend_kwargs

    logger.info(
        f"Colocated mode: creating {train_num_gpus} TrainWorkers "
        f"(TP={train_tp_size}, SP_ulysses={train_sp_ulysses_size}, "
        f"SP_ring={train_sp_ring_size})"
    )

    train_group = TrainWorkerGroup(
        num_workers=train_num_gpus,
        tp_size=train_tp_size,
        sp_ulysses_size=train_sp_ulysses_size,
        sp_ring_size=train_sp_ring_size,
        draft_model_args=draft_model_args,
        train_hparams=train_hparams_coloc,
        output_dir=args.output_dir,
        ckpt_dir=getattr(args, "ckpt_dir", None),
    )
    return None, train_group


def _build_disaggregated_groups(
    args,
    sglang_backend_kwargs,
    capture_layer_ids,
    draft_model_args,
    train_hparams,
    rollout_tp_size,
    train_tp_size,
    train_sp_ulysses_size,
    train_sp_ring_size,
) -> Tuple[RolloutWorkerGroup, TrainWorkerGroup]:
    """Create RolloutWorkerGroup + TrainWorkerGroup for disaggregated mode."""
    rollout_num_gpus = args.rollout_num_gpus
    train_num_gpus = args.train_num_gpus
    total_gpus = rollout_num_gpus + train_num_gpus

    logger.info(
        f"Disaggregated mode: "
        f"{rollout_num_gpus} rollout GPUs (TP={rollout_tp_size}), "
        f"{train_num_gpus} train GPUs "
        f"(TP={train_tp_size}, SP_ulysses={train_sp_ulysses_size}, "
        f"SP_ring={train_sp_ring_size})"
    )

    from ray.util.placement_group import placement_group

    bundles = [{"GPU": 1, "CPU": 1} for _ in range(total_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    ray.get(pg.ready())
    logger.info(
        f"Placement group ready: {total_gpus} bundles "
        f"(rollout={rollout_num_gpus}, train={train_num_gpus})"
    )

    transfer_backend = getattr(args, "transfer_backend", "ray")
    rollout_ranks = list(range(rollout_num_gpus))
    train_ranks = list(range(rollout_num_gpus, total_gpus))

    nccl_kwargs = {}
    if transfer_backend == "nccl":
        from specforge.distributed_ray import get_free_port as _get_free_port

        global_master_addr = _get_head_node_ip()
        global_master_port = _get_free_port()
        nccl_kwargs = dict(
            transfer_backend="nccl",
            global_world_size=total_gpus,
            global_master_addr=global_master_addr,
            global_master_port=global_master_port,
        )
        logger.info(
            f"NCCL transfer: global_world_size={total_gpus}, "
            f"rollout_ranks={rollout_ranks}, train_ranks={train_ranks}"
        )

    rollout_group = RolloutWorkerGroup(
        num_workers=rollout_num_gpus,
        tp_size=rollout_tp_size,
        target_model_path=args.target_model_path,
        backend=getattr(args, "target_model_backend", "sglang"),
        sglang_backend_kwargs=sglang_backend_kwargs,
        capture_layer_ids=capture_layer_ids,
        method=getattr(args, "method", "eagle3"),
        is_vlm=getattr(args, "is_vlm", False),
        torch_dtype="bfloat16",
        trust_remote_code=getattr(args, "trust_remote_code", False),
        model_download_dir=getattr(args, "model_download_dir", None),
        timeout_minutes=getattr(args, "dist_timeout", 20),
        placement_group=pg,
        placement_group_bundle_offset=0,
        global_rank_offset=0,
        rollout_ranks=rollout_ranks,
        **nccl_kwargs,
    )

    train_hparams_disagg = dict(train_hparams)
    train_hparams_disagg["disaggregate"] = True

    train_group = TrainWorkerGroup(
        num_workers=train_num_gpus,
        tp_size=train_tp_size,
        sp_ulysses_size=train_sp_ulysses_size,
        sp_ring_size=train_sp_ring_size,
        draft_model_args=draft_model_args,
        train_hparams=train_hparams_disagg,
        output_dir=args.output_dir,
        ckpt_dir=getattr(args, "ckpt_dir", None),
        placement_group=pg,
        placement_group_bundle_offset=rollout_num_gpus,
        global_rank_offset=rollout_num_gpus,
        train_ranks=train_ranks,
        **nccl_kwargs,
    )

    if transfer_backend == "nccl":
        logger.info("Waiting for all actors to complete global NCCL init ...")
        rollout_group.wait_ready()
        train_group.wait_ready()
        logger.info("All actors ready (NCCL global group initialized).")

    return rollout_group, train_group


def build_worker_groups(
    args,
    sglang_backend_kwargs: dict,
    capture_layer_ids,
) -> Tuple[Optional[RolloutWorkerGroup], TrainWorkerGroup]:
    """Create RolloutWorkerGroup and TrainWorkerGroup based on args."""
    disaggregate = getattr(args, "disaggregate", False)
    rollout_tp_size = getattr(args, "rollout_tp_size", 1)
    train_tp_size = getattr(args, "train_tp_size", 1)
    train_sp_ulysses_size = getattr(args, "train_sp_ulysses_size", 1)
    train_sp_ring_size = getattr(args, "train_sp_ring_size", 1)

    draft_model_args, train_hparams = _build_common_args(args, capture_layer_ids)

    if not disaggregate:
        return _build_colocated_groups(
            args,
            sglang_backend_kwargs,
            draft_model_args,
            train_hparams,
            train_tp_size,
            train_sp_ulysses_size,
            train_sp_ring_size,
        )
    else:
        return _build_disaggregated_groups(
            args,
            sglang_backend_kwargs,
            capture_layer_ids,
            draft_model_args,
            train_hparams,
            rollout_tp_size,
            train_tp_size,
            train_sp_ulysses_size,
            train_sp_ring_size,
        )
