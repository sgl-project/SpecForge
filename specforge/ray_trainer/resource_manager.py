"""
Resource manager for Ray-based Eagle3 training.

Manages placement groups and worker actor lifecycle for both
RolloutWorkerGroup (target model inference) and TrainWorkerGroup
(draft model training).
"""

import logging
import socket
from typing import List, Optional, Tuple

import ray

logger = logging.getLogger(__name__)


def _get_head_node_ip() -> str:
    """Return the IP address of the current (driver/head) node."""
    return socket.gethostbyname(socket.gethostname())


class RolloutWorkerGroup:
    """
    Manages a pool of RolloutWorker Ray Actors.

    All workers within the same TP group share a rendezvous port and form
    a single torch.distributed process group on the rollout side.  Only
    the tp_rank=0 worker in each TP group returns a populated RolloutBatch;
    the others participate in collective ops but return None.

    GPU allocation
    --------------
    * Each RolloutWorker is decorated with ``@ray.remote(num_gpus=1)``.
    * In disaggregated mode each worker gets an exclusive physical GPU.
    * In colocated mode the fractional GPU allocation is handled by the
      placement group created in ``build_worker_groups()``.
    """

    def __init__(
        self,
        num_workers: int,
        tp_size: int,
        target_model_path: str,
        backend: str,
        sglang_backend_kwargs: dict,
        aux_hidden_states_layers,
        is_vlm: bool = False,
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        model_download_dir: Optional[str] = None,
        timeout_minutes: int = 20,
        placement_group=None,           # optional Ray PlacementGroup
        placement_group_bundle_offset: int = 0,
    ) -> None:
        from specforge.distributed_ray import get_free_port
        from specforge.ray_workers import RolloutWorker

        assert num_workers % tp_size == 0, (
            f"num_workers ({num_workers}) must be divisible by tp_size ({tp_size})"
        )

        self.num_workers = num_workers
        self.tp_size = tp_size
        self.num_tp_groups = num_workers // tp_size
        master_addr = _get_head_node_ip()

        # One rendezvous port per TP group
        master_ports = [get_free_port() for _ in range(self.num_tp_groups)]

        self._workers: List = []
        for rank in range(num_workers):
            tp_group_idx = rank // tp_size
            tp_rank = rank % tp_size

            if placement_group is not None:
                bundle_index = placement_group_bundle_offset + rank
                actor = RolloutWorker.options(
                    placement_group=placement_group,
                    placement_group_bundle_index=bundle_index,
                ).remote(
                    rank=tp_rank,
                    world_size=tp_size,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    master_addr=master_addr,
                    master_port=master_ports[tp_group_idx],
                    target_model_path=target_model_path,
                    backend=backend,
                    sglang_backend_kwargs=sglang_backend_kwargs,
                    aux_hidden_states_layers=aux_hidden_states_layers,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code,
                    model_download_dir=model_download_dir,
                    is_vlm=is_vlm,
                    timeout_minutes=timeout_minutes,
                )
            else:
                actor = RolloutWorker.remote(
                    rank=tp_rank,
                    world_size=tp_size,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    master_addr=master_addr,
                    master_port=master_ports[tp_group_idx],
                    target_model_path=target_model_path,
                    backend=backend,
                    sglang_backend_kwargs=sglang_backend_kwargs,
                    aux_hidden_states_layers=aux_hidden_states_layers,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code,
                    model_download_dir=model_download_dir,
                    is_vlm=is_vlm,
                    timeout_minutes=timeout_minutes,
                )
            self._workers.append(actor)

        # Wait for all workers to finish initialisation
        ray.get([w.is_ready.remote() for w in self._workers])
        logger.info(f"RolloutWorkerGroup: {num_workers} workers ready.")

    def generate_rollout_batch(self, data_batch: dict):
        """
        Distribute *data_batch* to all TP groups and invoke rollout in parallel.

        Each TP group processes the full ``data_batch`` (the target model is
        TP-sharded across the group).  Only the tp_rank=0 worker in each
        group returns a populated RolloutBatch; other workers return None.

        Returns
        -------
        List of Ray ObjectRefs – one per TP group.  Callers should
        ``ray.get()`` to obtain the actual ``RolloutBatch``.
        """
        refs = []
        for tp_group_idx in range(self.num_tp_groups):
            # Broadcast to all workers in this TP group
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

    def shutdown(self) -> None:
        """Shutdown all workers and destroy placement group if owned."""
        ray.get([w.shutdown.remote() for w in self._workers])
        logger.info("RolloutWorkerGroup shutdown complete.")


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
    ) -> None:
        from specforge.distributed_ray import get_free_port
        from specforge.ray_workers import TrainWorker

        sp_size = sp_ulysses_size * sp_ring_size
        assert num_workers % tp_size == 0, (
            f"num_workers ({num_workers}) must be divisible by tp_size ({tp_size})"
        )
        assert num_workers % sp_size == 0, (
            f"num_workers ({num_workers}) must be divisible by sp_size ({sp_size})"
        )

        self.num_workers = num_workers
        self.tp_size = tp_size
        self.sp_ulysses_size = sp_ulysses_size
        self.sp_ring_size = sp_ring_size
        master_addr = _get_head_node_ip()
        master_port = get_free_port()
        print('4444')
        self._workers: List = []
        for rank in range(num_workers):
            if placement_group is not None:
                bundle_index = placement_group_bundle_offset + rank
                actor = TrainWorker.options(
                    placement_group=placement_group,
                    placement_group_bundle_index=bundle_index,
                ).remote(
                    rank=rank,
                    world_size=num_workers,
                    tp_size=tp_size,
                    sp_ulysses_size=sp_ulysses_size,
                    sp_ring_size=sp_ring_size,
                    master_addr=master_addr,
                    master_port=master_port,
                    output_dir=output_dir,
                    ckpt_dir=ckpt_dir,
                    **draft_model_args,
                    **train_hparams,
                )
            else:
                actor = TrainWorker.remote(
                    rank=rank,
                    world_size=num_workers,
                    tp_size=tp_size,
                    sp_ulysses_size=sp_ulysses_size,
                    sp_ring_size=sp_ring_size,
                    master_addr=master_addr,
                    master_port=master_port,
                    output_dir=output_dir,
                    ckpt_dir=ckpt_dir,
                    **draft_model_args,
                    **train_hparams,
                )
            self._workers.append(actor)
            print('555')
        ray.get([w.is_ready.remote() for w in self._workers])
        print('666')
        logger.info(f"TrainWorkerGroup: {num_workers} workers ready.")

    def get_dataset_info(self) -> dict:
        """Return dataset info from rank-0 worker."""
        return ray.get(self._workers[0].get_dataset_info.remote())

    def set_epoch(self, epoch: int) -> None:
        """Set epoch on all workers (for DistributedSampler)."""
        ray.get([w.set_epoch.remote(epoch) for w in self._workers])

    def train_step(self, rollout_batch_refs: list, global_step: int) -> dict:
        """
        Broadcast the rollout_batch_ref to all workers and execute train_step.

        All workers receive the same full RolloutBatch and internally
        shard it by tp_rank.  SP partitioning is handled by UspAdapter.

        Returns the metrics dict from rank-0.
        """
        # We only need one ref (the full batch)
        rollout_ref = rollout_batch_refs[0] if rollout_batch_refs else None

        step_refs = [
            w.run_step.remote(global_step) for w in self._workers
        ]
        results = ray.get(step_refs)
        # rank-0 returns the authoritative metrics
        return results[0]

    def eval_step(self) -> dict:
        """Run eval across all workers and return rank-0 metrics."""
        eval_refs = [w.run_eval_step.remote() for w in self._workers]
        results = ray.get(eval_refs)
        return results[0]

    def save_checkpoint(self, epoch: int, step: int) -> str:
        """Save checkpoint; returns path from rank-0."""
        ckpt_refs = [w.save_checkpoint.remote(epoch, step) for w in self._workers]
        results = ray.get(ckpt_refs)
        return results[0]  # rank-0 returns the path

    def shutdown(self) -> None:
        """Shutdown all workers."""
        ray.get([w.shutdown.remote() for w in self._workers])
        logger.info("TrainWorkerGroup shutdown complete.")


def build_worker_groups(
    args,
    sglang_backend_kwargs: dict,
    aux_hidden_states_layers,
) -> Tuple[Optional[RolloutWorkerGroup], TrainWorkerGroup]:
    """
    Create RolloutWorkerGroup and TrainWorkerGroup based on args.

    Colocated mode (disaggregate=False)
    ------------------------------------
    * A single placement group with PACK strategy is created so that
      each RolloutWorker and its paired TrainWorker share the same
      physical GPU.
    * Each bundle requests 1 GPU split between rollout (0.5) and
      train (0.5) fractional allocation.

    Disaggregated mode (disaggregate=True)
    ----------------------------------------
    * Two separate placement groups are created with SPREAD strategy
      to isolate rollout and train GPUs.

    Returns
    -------
    (rollout_group, train_group)
    In colocated mode rollout is handled internally by TrainWorker,
    so rollout_group is returned as None.
    """
    disaggregate = getattr(args, "disaggregate", False)
    rollout_tp_size = getattr(args, "rollout_tp_size", 1)
    train_tp_size = getattr(args, "train_tp_size", 1)
    train_sp_ulysses_size = getattr(args, "train_sp_ulysses_size", 1)
    train_sp_ring_size = getattr(args, "train_sp_ring_size", 1)

    # Build draft model args dict
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
    )

    train_hparams = dict(
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=getattr(args, "total_steps", None) or 0,
        ttt_length=args.ttt_length,
        draft_accumulation_steps=args.draft_accumulation_steps,
        resume=getattr(args, "resume", False),
        target_model_backend=getattr(args, "target_model_backend", "sglang"),
        sglang_backend_kwargs=sglang_backend_kwargs,
        aux_hidden_states_layers=aux_hidden_states_layers,
    )

    if not disaggregate:
        # ── Colocated mode ──────────────────────────────────────────────────
        # TrainWorkers load the target model locally; no separate RolloutWorkerGroup.
        train_num_gpus = getattr(args, "train_num_gpus", None)
        if train_num_gpus is None:
            # Auto-detect from CUDA_VISIBLE_DEVICES or ray cluster
            import torch
            train_num_gpus = torch.cuda.device_count()

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
            train_hparams=train_hparams,
            output_dir=args.output_dir,
            ckpt_dir=getattr(args, "ckpt_dir", None),
        )
        return None, train_group

    else:
        # ── Disaggregated mode ──────────────────────────────────────────────
        rollout_num_gpus = args.rollout_num_gpus
        train_num_gpus = args.train_num_gpus

        logger.info(
            f"Disaggregated mode: "
            f"{rollout_num_gpus} rollout GPUs (TP={rollout_tp_size}), "
            f"{train_num_gpus} train GPUs "
            f"(TP={train_tp_size}, SP_ulysses={train_sp_ulysses_size}, "
            f"SP_ring={train_sp_ring_size})"
        )

        # Rollout workers don't call the target model locally
        draft_model_args_disagg = dict(draft_model_args)
        draft_model_args_disagg["rollout_worker_ref"] = None  # will be set later
        # In disaggregated mode, TrainWorkers have no local target model
        train_hparams_disagg = dict(train_hparams)
        train_hparams_disagg["target_model_backend"] = "none"
        train_hparams_disagg["sglang_backend_kwargs"] = None
        train_hparams_disagg["aux_hidden_states_layers"] = None
        print('111')
        rollout_group = RolloutWorkerGroup(
            num_workers=rollout_num_gpus,
            tp_size=rollout_tp_size,
            target_model_path=args.target_model_path,
            backend=getattr(args, "target_model_backend", "sglang"),
            sglang_backend_kwargs=sglang_backend_kwargs,
            aux_hidden_states_layers=aux_hidden_states_layers,
            is_vlm=getattr(args, "is_vlm", False),
            torch_dtype="bfloat16",
            trust_remote_code=getattr(args, "trust_remote_code", False),
            model_download_dir=getattr(args, "model_download_dir", None),
            timeout_minutes=getattr(args, "dist_timeout", 20),
        )
        print('222')
        train_group = TrainWorkerGroup(
            num_workers=train_num_gpus,
            tp_size=train_tp_size,
            sp_ulysses_size=train_sp_ulysses_size,
            sp_ring_size=train_sp_ring_size,
            draft_model_args=draft_model_args_disagg,
            train_hparams=train_hparams_disagg,
            output_dir=args.output_dir,
            ckpt_dir=getattr(args, "ckpt_dir", None),
        )
        print('333')
        return rollout_group, train_group
