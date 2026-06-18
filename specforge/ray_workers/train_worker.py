"""
TrainWorker – Ray Actor that hosts the **draft model** and drives
the Eagle3 forward/backward training step.

One TrainWorker occupies one GPU.  Multiple TrainWorkers form a torch.distributed
process group and use FSDP for data-parallel gradient synchronisation.

Colocated vs. Disaggregated
----------------------------
* colocated   (disaggregate=False):
    The TrainWorker also loads the target model on the same GPU and runs
    rollout internally before the draft-model forward pass.
* disaggregated (disaggregate=True):
    The TrainWorker does NOT load the target model.  It receives a
    pre-computed RolloutBatch (as a Ray ObjectRef) from the pipeline,
    which obtained it from a separate RolloutWorkerGroup.
"""

import hashlib
import logging
import os
from typing import List, Optional

import ray
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class TrainWorker:
    """
    Hosts the Eagle3 draft model (FSDP-wrapped) and orchestrates one
    training step: data loading → rollout → forward → backward → update.

    All workers in the same TrainWorkerGroup share a torch.distributed
    process group and execute FSDP collective ops in lockstep.
    """

    # ─────────────────────────────────────────────────────────────────────
    # Construction
    # ─────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        rank: int,
        world_size: int,
        tp_size: int,
        sp_ulysses_size: int,
        sp_ring_size: int,
        master_addr: str,
        master_port: int,
        # model paths
        target_model_path: str,
        draft_model_config_path: Optional[str],
        attention_backend: str,
        embedding_key: str,
        # mode control
        disaggregate: bool = False,
        target_model_backend: str = "sglang",
        sglang_backend_kwargs: Optional[dict] = None,
        # dataset / dataloader
        train_data_path: Optional[str] = None,
        eval_data_path: Optional[str] = None,
        chat_template: str = "qwen",
        max_length: int = 4096,
        batch_size: int = 1,
        dataloader_num_workers: int = 4,
        build_dataset_num_proc: int = 8,
        is_preformatted: bool = False,
        train_only_last_turn: bool = False,
        is_vlm: bool = False,
        cache_dir: str = "./cache",
        # training hyper-params
        learning_rate: float = 1e-4,
        max_grad_norm: float = 0.5,
        warmup_ratio: float = 0.015,
        total_steps: int = 100_000,
        ttt_length: int = 7,
        draft_accumulation_steps: int = 1,
        log_interval: int = 50,
        # checkpoint
        ckpt_dir: Optional[str] = None,
        output_dir: str = "./output",
        resume: bool = False,
        # misc
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        model_download_dir: Optional[str] = None,
        timeout_minutes: int = 20,
        seed: int = 0,
        # NCCL transfer mode params
        transfer_backend: str = "ray",
        global_rank: Optional[int] = None,
        global_world_size: Optional[int] = None,
        global_master_addr: Optional[str] = None,
        global_master_port: Optional[int] = None,
        train_ranks: Optional[list] = None,
        # Method dispatch
        method: str = "eagle3",
        capture_layer_ids: Optional[list] = None,
        # DFlash-specific
        block_size: int = 16,
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        mask_token_id: Optional[int] = None,
        lm_head_key: str = "lm_head.weight",
        num_draft_layers: int = 5,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.tp_size = tp_size
        self.sp_ulysses_size = sp_ulysses_size
        self.sp_ring_size = sp_ring_size
        self.sp_size = sp_ulysses_size * sp_ring_size
        self.attention_backend = attention_backend
        self.ttt_length = ttt_length
        self.draft_accumulation_steps = draft_accumulation_steps
        self._log_interval = log_interval
        self.output_dir = output_dir
        self.is_vlm = is_vlm
        self.batch_size = batch_size
        self.target_batch_size = tp_size * batch_size
        self.max_length = max_length
        self._disaggregate = disaggregate
        self._draft_model_config_path = draft_model_config_path
        self._draft_model_config = None
        self._ready = False
        self._transfer_backend = transfer_backend
        self._global_rank = global_rank
        self._method = method

        # adjust accumulation for SP (mirrors sp_sanity_check in train_eagle3.py)
        if self.sp_size > 1:
            self.draft_accumulation_steps *= self.sp_size

        # ── 1. Build dataset BEFORE CUDA init ─────────────────────────────
        # HuggingFace datasets.map(num_proc=N) uses fork-based multiprocessing.
        # CUDA does not support fork after context init, so we must build
        # the dataset before any CUDA/NCCL calls.
        self._train_eagle3_dataset = None
        self._eval_eagle3_dataset = None
        self._vocab_mapping_path = None
        self._train_iter = None
        self._prefetched_batch = None
        self._nccl_cached_splits = None

        if train_data_path is not None:
            self._prepare_datasets(
                train_data_path=train_data_path,
                eval_data_path=eval_data_path,
                chat_template=chat_template,
                max_length=max_length,
                cache_dir=cache_dir,
                build_dataset_num_proc=build_dataset_num_proc,
                is_preformatted=is_preformatted,
                train_only_last_turn=train_only_last_turn,
                target_model_path=target_model_path,
                trust_remote_code=trust_remote_code,
            )

        # ── 2. Init distributed (CUDA + NCCL) ─────────────────────────────
        if transfer_backend == "nccl":
            from specforge.distributed_ray import (
                init_global_distributed,
                init_train_subgroup,
            )

            init_global_distributed(
                global_rank=global_rank,
                global_world_size=global_world_size,
                master_addr=global_master_addr,
                master_port=global_master_port,
                timeout_minutes=timeout_minutes,
            )
            init_train_subgroup(
                train_ranks=train_ranks,
                tp_size=tp_size,
                sp_ulysses_size=sp_ulysses_size,
                sp_ring_size=sp_ring_size,
            )
        else:
            from specforge.distributed_ray import init_train_distributed

            init_train_distributed(
                rank=rank,
                world_size=world_size,
                master_addr=master_addr,
                master_port=master_port,
                tp_size=tp_size,
                sp_ulysses_size=sp_ulysses_size,
                sp_ring_size=sp_ring_size,
                timeout_minutes=timeout_minutes,
            )
        from specforge.distributed import (
            get_dp_group,
            get_draft_dp_rank,
            get_draft_sp_group,
            get_sp_rank,
            get_sp_size,
            get_tp_group,
        )

        self._tp_group = get_tp_group()
        self._dp_group = get_dp_group()
        self._tp_rank = dist.get_rank(self._tp_group)
        self._sp_group = get_draft_sp_group()
        self._sp_rank = get_sp_rank()
        self._sp_size = get_sp_size()
        self._draft_dp_rank = get_draft_dp_rank()

        # ── 2. Load draft model ────────────────────────────────────────────
        from specforge.utils import get_last_checkpoint

        ckpt_info = (0, 0)
        resume_state = None
        draft_model_last_checkpoint = None
        is_resume_checkpoint = False

        if ckpt_dir is not None and os.path.isdir(ckpt_dir):
            draft_model_last_checkpoint = ckpt_dir

        if resume and os.path.isdir(output_dir):
            draft_model_last_checkpoint, ckpt_info = get_last_checkpoint(output_dir)
            is_resume_checkpoint = True

        if method == "eagle3":
            from specforge.modeling import AutoDraftModelConfig, AutoEagle3DraftModel
            from specforge.utils import create_draft_config_from_target

            if draft_model_config_path is None:
                auto_cfg_path = create_draft_config_from_target(
                    target_model_path=target_model_path,
                    cache_dir=model_download_dir,
                )
                self._draft_model_config = AutoDraftModelConfig.from_file(auto_cfg_path)
            else:
                self._draft_model_config = AutoDraftModelConfig.from_file(
                    draft_model_config_path
                )

            if draft_model_last_checkpoint and not is_resume_checkpoint:
                self._draft_model_config = AutoDraftModelConfig.from_file(
                    os.path.join(draft_model_last_checkpoint, "config.json")
                )

            if draft_model_last_checkpoint:
                self._draft_model = AutoEagle3DraftModel.from_pretrained(
                    draft_model_last_checkpoint,
                    attention_backend=attention_backend,
                    torch_dtype=getattr(torch, torch_dtype),
                ).cuda()
            else:
                self._draft_model = AutoEagle3DraftModel.from_config(
                    self._draft_model_config,
                    attention_backend=attention_backend,
                    torch_dtype=getattr(torch, torch_dtype),
                ).cuda()

            self._draft_model.load_embedding(
                target_model_path, embedding_key=embedding_key
            )
            self._draft_model.freeze_embedding()

        elif method == "dflash":
            from transformers import AutoConfig, AutoTokenizer

            from specforge.modeling.draft.dflash import DFlashDraftModel

            if draft_model_config_path is not None:
                draft_config = AutoConfig.from_pretrained(draft_model_config_path)
            else:
                import copy

                target_config = AutoConfig.from_pretrained(target_model_path)
                draft_config = copy.deepcopy(target_config)
                draft_config.num_hidden_layers = num_draft_layers
                draft_config.block_size = block_size
                draft_config.num_target_layers = target_config.num_hidden_layers

            if (
                not hasattr(draft_config, "dflash_config")
                or draft_config.dflash_config is None
            ):
                draft_config.dflash_config = {}

            draft_config._attn_implementation = attention_backend
            self._draft_model_config = draft_config

            if draft_model_last_checkpoint and not is_resume_checkpoint:
                self._draft_model_config = AutoConfig.from_pretrained(
                    os.path.join(draft_model_last_checkpoint, "config.json")
                )

            if draft_model_last_checkpoint:
                self._draft_model = DFlashDraftModel.from_pretrained(
                    draft_model_last_checkpoint,
                    torch_dtype=getattr(torch, torch_dtype),
                ).cuda()
            else:
                self._draft_model = (
                    DFlashDraftModel(draft_config)
                    .cuda()
                    .to(getattr(torch, torch_dtype))
                )

            # Resolve mask_token_id
            if mask_token_id is None:
                tokenizer = AutoTokenizer.from_pretrained(target_model_path)
                mask_token_id = tokenizer.mask_token_id or 151669  # fallback
            self._draft_model.mask_token_id = mask_token_id
            self._draft_model.config.dflash_config["mask_token_id"] = mask_token_id
            self._draft_model.config.dflash_config["target_layer_ids"] = (
                self._draft_model.target_layer_ids
            )

        if is_resume_checkpoint and draft_model_last_checkpoint:
            training_state_path = os.path.join(
                draft_model_last_checkpoint, "training_state.pt"
            )
            if os.path.exists(training_state_path):
                resume_state = torch.load(
                    training_state_path, map_location="cpu", weights_only=False
                )

        # ── 3. Colocated only: load target model ─────────────────────────
        self._target_model = None
        if not self._disaggregate:
            dtype = getattr(torch, torch_dtype)
            if method == "eagle3":
                from specforge.modeling.target import get_eagle3_target_model

                if sglang_backend_kwargs is None:
                    sglang_backend_kwargs = {}
                self._target_model = get_eagle3_target_model(
                    pretrained_model_name_or_path=target_model_path,
                    backend=target_model_backend,
                    torch_dtype=dtype,
                    device="cuda",
                    cache_dir=model_download_dir,
                    trust_remote_code=trust_remote_code,
                    **sglang_backend_kwargs,
                )
                self._target_model.set_aux_hidden_states_layers(capture_layer_ids)
            elif method == "dflash":
                from specforge.modeling.target.dflash_target_model import (
                    get_dflash_target_model,
                )

                if sglang_backend_kwargs is None:
                    sglang_backend_kwargs = {}
                self._target_model = get_dflash_target_model(
                    pretrained_model_name_or_path=target_model_path,
                    backend=target_model_backend,
                    torch_dtype=dtype,
                    device="cuda" if target_model_backend == "hf" else None,
                    cache_dir=model_download_dir,
                    trust_remote_code=trust_remote_code,
                    **sglang_backend_kwargs,
                )
                if capture_layer_ids is not None:
                    self._target_model.set_capture_layers(capture_layer_ids)

        # ── 4. Build Online Model + FSDP ─────────────────────────────────
        if method == "eagle3":
            from specforge.core import OnlineEagle3Model

            online_model = OnlineEagle3Model(
                draft_model=self._draft_model,
                length=ttt_length,
                attention_backend=attention_backend,
            )
        elif method == "dflash":
            from specforge.core.dflash import OnlineDFlashModel
            from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead

            target_components = TargetEmbeddingsAndHead.from_pretrained(
                target_model_path,
                embed_key=embedding_key,
                lm_head_key=lm_head_key,
                device="cuda",
                trust_remote_code=trust_remote_code,
            )
            online_model = OnlineDFlashModel(
                draft_model=self._draft_model,
                target_lm_head=target_components.lm_head,
                target_embed_tokens=target_components.embed_tokens,
                block_size=block_size,
                mask_token_id=mask_token_id,
                attention_backend=attention_backend,
                num_anchors=num_anchors,
                loss_decay_gamma=loss_decay_gamma,
            )

        self._online_model = FSDP(
            online_model,
            use_orig_params=True,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            process_group=self._dp_group,
        )

        # ── 5. Optimizer ──────────────────────────────────────────────────
        from specforge.optimizer import BF16Optimizer

        self._optimizer = BF16Optimizer(
            self._draft_model,
            lr=learning_rate,
            max_grad_norm=max_grad_norm,
            warmup_ratio=warmup_ratio,
            total_steps=total_steps,
        )
        if resume_state is not None:
            self._optimizer.load_state_dict(resume_state)

        self._start_epoch = ckpt_info[0]
        self._global_step = ckpt_info[1]

        # ── 6. Build DataLoaders from pre-built datasets ─────────────────
        self._train_dataloader = None
        self._eval_dataloader = None
        self._train_iter = None

        if self._train_eagle3_dataset is not None:
            self._build_dataloaders(
                dataloader_num_workers=dataloader_num_workers,
            )

        self._ready = True
        logger.info(f"TrainWorker rank={rank} ready (disaggregate={disaggregate}).")

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _prepare_datasets(
        self,
        train_data_path: str,
        eval_data_path: Optional[str],
        chat_template: str,
        max_length: int,
        cache_dir: str,
        build_dataset_num_proc: int,
        is_preformatted: bool,
        train_only_last_turn: bool,
        target_model_path: str,
        trust_remote_code: bool,
    ) -> None:
        """
        Build datasets and vocab mapping BEFORE CUDA initialisation.

        HuggingFace datasets.map(num_proc=N) uses fork-based multiprocessing.
        CUDA does not support fork after context init, so all CPU-heavy
        dataset processing must happen here, before init_train_distributed().
        """
        from transformers import AutoTokenizer

        from datasets import Dataset
        from specforge.data import build_eagle3_dataset, generate_vocab_mapping_file
        from specforge.modeling import AutoDraftModelConfig
        from specforge.utils import safe_conversations_generator

        tokenizer = AutoTokenizer.from_pretrained(
            target_model_path, trust_remote_code=trust_remote_code
        )

        cache_params_string = (
            f"{train_data_path}-{max_length}-{chat_template}-{target_model_path}"
        )
        cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

        train_dataset_raw = Dataset.from_generator(
            generator=safe_conversations_generator,
            gen_kwargs={"file_path": train_data_path},
        )

        # Note: rank_0_priority requires dist to be initialised, but we
        # haven't called init_train_distributed yet.  Since all Ray actors
        # build the dataset independently (with caching), we can skip the
        # rank_0_priority wrapper here — the cache ensures no redundant work.
        self._train_eagle3_dataset = build_eagle3_dataset(
            dataset=train_dataset_raw,
            tokenizer=tokenizer,
            chat_template=chat_template,
            max_length=max_length,
            cache_dir=os.path.join(cache_dir, "processed_dataset"),
            cache_key=cache_key,
            is_vlm=self.is_vlm,
            is_preformatted=is_preformatted,
            num_proc=build_dataset_num_proc,
            train_only_last_turn=train_only_last_turn,
        )

        # Read draft model config for vocab sizes (needed for vocab mapping)
        # DFlash uses full target vocab, so no vocab mapping needed.
        if self._method == "eagle3":
            if (
                not hasattr(self, "_draft_model_config")
                or self._draft_model_config is None
            ):
                from specforge.utils import create_draft_config_from_target

                if self._draft_model_config_path is None:
                    auto_cfg_path = create_draft_config_from_target(
                        target_model_path=target_model_path,
                    )
                    self._draft_model_config = AutoDraftModelConfig.from_file(
                        auto_cfg_path
                    )
                else:
                    self._draft_model_config = AutoDraftModelConfig.from_file(
                        self._draft_model_config_path
                    )

            self._vocab_mapping_path = generate_vocab_mapping_file(
                dataset=self._train_eagle3_dataset,
                target_vocab_size=self._draft_model_config.vocab_size,
                draft_vocab_size=self._draft_model_config.draft_vocab_size,
                cache_dir=os.path.join(cache_dir, "vocab_mapping"),
                cache_key=cache_key,
            )
        else:
            self._vocab_mapping_path = None

        if eval_data_path is not None:
            eval_dataset_raw = Dataset.from_generator(
                generator=safe_conversations_generator,
                gen_kwargs={"file_path": eval_data_path},
            )
            self._eval_eagle3_dataset = build_eagle3_dataset(
                eval_dataset_raw,
                tokenizer,
                chat_template,
                max_length,
                is_vlm=self.is_vlm,
                num_proc=build_dataset_num_proc,
                is_preformatted=is_preformatted,
                train_only_last_turn=train_only_last_turn,
            )

    def _build_dataloaders(
        self,
        dataloader_num_workers: int,
    ) -> None:
        """
        Create DataLoaders from pre-built datasets.

        Must be called AFTER init_train_distributed() because it needs
        the DP/SP process groups for DistributedSampler.
        """
        from specforge.data import prepare_dp_dataloaders
        from specforge.distributed import get_dp_group, get_draft_dp_group

        if self._vocab_mapping_path is not None:
            self._draft_model.load_vocab_mapping(self._vocab_mapping_path)

        sampler_group = get_draft_dp_group() if self.sp_size > 1 else get_dp_group()

        self._train_dataloader = prepare_dp_dataloaders(
            self._train_eagle3_dataset,
            self.target_batch_size,
            num_workers=dataloader_num_workers,
            shuffle=True,
            process_group=sampler_group,
            is_vlm=self.is_vlm,
        )

        if self._eval_eagle3_dataset is not None:
            self._eval_dataloader = prepare_dp_dataloaders(
                self._eval_eagle3_dataset,
                self.target_batch_size,
                num_workers=dataloader_num_workers,
                shuffle=False,
                process_group=sampler_group,
                is_vlm=self.is_vlm,
            )

        # Free dataset references — DataLoader holds its own copy
        self._train_eagle3_dataset = None
        self._eval_eagle3_dataset = None

    def _local_rollout(self, data: dict) -> "RolloutBatch":
        """
        Generate RolloutBatch using the locally loaded target model.
        Only used in colocated mode.  Tensors stay on GPU (no CPU round-trip).
        """
        from specforge.ray_workers.worker_utils import RolloutBatch

        if self._method == "eagle3":
            output = self._target_model.generate_eagle3_data(
                input_ids=data["input_ids"].cuda(),
                attention_mask=data["attention_mask"].cuda(),
                loss_mask=data["loss_mask"].cuda(),
            )
            return RolloutBatch(
                input_ids=output.input_ids,
                attention_mask=output.attention_mask,
                loss_mask=output.loss_mask,
                hidden_states=output.hidden_states,
                target=output.target,
            )
        elif self._method == "dflash":
            output = self._target_model.generate_dflash_data(
                input_ids=data["input_ids"].cuda(),
                attention_mask=data["attention_mask"].cuda(),
                loss_mask=data["loss_mask"].cuda(),
            )
            return RolloutBatch(
                input_ids=output.input_ids,
                attention_mask=output.attention_mask,
                loss_mask=output.loss_mask,
                hidden_states=output.hidden_states,
                target=None,
            )

    def _forward(self, rollout_batch: "RolloutBatch"):
        """Run the online model forward with pre-computed rollout data."""
        from specforge.ray_workers.worker_utils import (
            batch_shard_by_tp,
            batch_to_device,
        )

        device = torch.device("cuda")
        rb = batch_to_device(rollout_batch, device)
        rb = batch_shard_by_tp(rb, self.tp_size, self._tp_rank)

        if self._method == "eagle3":
            plosses, _, acces = self._online_model(
                input_ids=rb.input_ids,
                attention_mask=rb.attention_mask,
                loss_mask=rb.loss_mask,
                target=rb.target,
                hidden_states=rb.hidden_states,
                position_ids=rb.position_ids,
            )
            return plosses, acces
        elif self._method == "dflash":
            loss, accuracy = self._online_model(
                input_ids=rb.input_ids,
                hidden_states=rb.hidden_states,
                loss_mask=rb.loss_mask,
            )
            # Wrap in lists for uniform interface with Eagle3
            return [loss], [accuracy]

    def _metrics_dict(
        self,
        plosses: List[torch.Tensor],
        acces: List[torch.Tensor],
        mode: str = "train",
        should_log: bool = True,
    ) -> dict:
        """Reduce metrics across DP and return a plain Python dict.

        When *should_log* is False, skips the expensive all_reduce and
        returns an empty dict.  This avoids a cross-GPU synchronization
        point on non-logging steps.
        """
        if not should_log:
            return {}
        accuracies = torch.stack(acces)
        losses = torch.stack(plosses)
        dist.all_reduce(accuracies, op=dist.ReduceOp.AVG, group=self._dp_group)
        dist.all_reduce(losses, op=dist.ReduceOp.AVG, group=self._dp_group)
        metrics = {f"{mode}/lr": self._optimizer.get_learning_rate()}
        for i, (a, l) in enumerate(
            zip(accuracies.cpu().tolist(), losses.cpu().tolist())
        ):
            metrics[f"{mode}/acc_{i}"] = a
            metrics[f"{mode}/ploss_{i}"] = l
        return metrics

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._ready

    def get_dataset_info(self) -> dict:
        """Return dataset sizes so the orchestrator can compute total_steps."""
        train_size = (
            len(self._train_dataloader) if self._train_dataloader is not None else 0
        )
        eval_size = (
            len(self._eval_dataloader) if self._eval_dataloader is not None else 0
        )
        return {
            "train_steps_per_epoch": train_size,
            "eval_steps": eval_size,
            "start_epoch": self._start_epoch,
            "global_step": self._global_step,
        }

    def set_epoch(self, epoch: int) -> None:
        """Update the DistributedSampler epoch (call at start of each epoch)."""
        if self._train_dataloader is not None:
            self._train_dataloader.sampler.set_epoch(epoch + 1)
        self._train_iter = (
            iter(self._train_dataloader) if self._train_dataloader else None
        )
        self._prefetched_batch = None
        self._nccl_cached_splits = None

    def fetch_batch(self) -> Optional[dict]:
        """
        Advance the DataLoader iterator and return the raw data batch.

        Used by the pipeline in disaggregated mode to obtain input data
        for the RolloutWorkerGroup.  Only rank-0's result is used by the
        pipeline; all other ranks still advance their iterators to keep
        the DistributedSampler in sync.

        Returns None if the iterator is exhausted or not initialised.
        """
        if self._train_iter is None:
            return None
        try:
            data = next(self._train_iter)
            # Store for run_step to skip re-fetching
            self._prefetched_batch = data
            # Move to CPU dict of tensors for Ray serialisation
            return {k: v.cpu() if hasattr(v, "cpu") else v for k, v in data.items()}
        except StopIteration:
            return None

    def run_step(
        self,
        global_step: int,
        rollout_batch_ref=None,
        split_index: int = 0,
        split_count: int = 1,
        nccl_src_rank: int = -1,
        skip: bool = False,
    ) -> Optional[dict]:
        """
        Execute one training step.

        Args:
            global_step: Current global training step.
            rollout_batch_ref: (disaggregated mode only) Ray ObjectRef pointing
                to a pre-computed RolloutBatch from RolloutWorkerGroup.
                None in colocated mode — rollout is done locally.
            skip: If True, advance the DataLoader iterator without training
                (used when resuming mid-epoch).

        Returns:
            Metrics dict on every rank (orchestrator takes rank-0's copy).
        """
        # In disaggregated mode with external rollout data, skip local
        # DataLoader read — the driver feeds data directly to RolloutWorkers.
        has_nccl_cache = nccl_src_rank == -1 and self._nccl_cached_splits is not None
        if self._disaggregate and (
            rollout_batch_ref is not None or nccl_src_rank >= 0 or has_nccl_cache
        ):
            data = None
        elif self._train_iter is None:
            raise RuntimeError("Call set_epoch() before run_step().")
        else:
            # Use prefetched batch from fetch_batch() if available,
            # otherwise fetch from the iterator directly.
            if self._prefetched_batch is not None:
                data = self._prefetched_batch
                self._prefetched_batch = None
            else:
                data = next(self._train_iter)

        if skip:
            return None

        self._draft_model.train()

        # Obtain RolloutBatch
        if nccl_src_rank >= 0:
            # NCCL mode: SP leader receives from RolloutWorker,
            # then broadcasts to SP group members.
            from specforge.ray_workers.worker_utils import (
                batch_split,
                nccl_broadcast_rollout_batch,
                nccl_recv_rollout_batch,
            )

            if self._sp_size > 1:
                # Multi-GPU SP: leader recvs, then broadcasts
                if self._sp_rank == 0:
                    rollout_batch = nccl_recv_rollout_batch(
                        src_rank=nccl_src_rank,
                        device=torch.device("cuda"),
                    )
                    rollout_batch = nccl_broadcast_rollout_batch(
                        rollout_batch,
                        src=0,
                        group=self._sp_group,
                    )
                else:
                    rollout_batch = nccl_broadcast_rollout_batch(
                        None,
                        src=0,
                        group=self._sp_group,
                        device=torch.device("cuda"),
                    )
            else:
                # Single-GPU SP (DP=1 or no SP): direct recv
                rollout_batch = nccl_recv_rollout_batch(
                    src_rank=nccl_src_rank,
                    device=torch.device("cuda"),
                )
            # Cache for subsequent splits
            if split_count > 1:
                self._nccl_cached_splits = batch_split(rollout_batch, split_count)
                rollout_batch = self._nccl_cached_splits[split_index]
            else:
                self._nccl_cached_splits = None
        elif nccl_src_rank == -1 and self._nccl_cached_splits is not None:
            # NCCL mode: use cached split from previous recv
            rollout_batch = self._nccl_cached_splits[split_index]
        elif rollout_batch_ref is not None:
            # Disaggregated mode: use pre-computed batch from RolloutWorkerGroup.
            # Ray auto-resolves ObjectRefs passed as remote() arguments,
            # so rollout_batch_ref may already be a RolloutBatch.
            if isinstance(rollout_batch_ref, ray.ObjectRef):
                rollout_batch = ray.get(rollout_batch_ref)
            else:
                rollout_batch = rollout_batch_ref
            # Split if this is a multi-batch rollout result
            if split_count > 1:
                from specforge.ray_workers.worker_utils import batch_split

                splits = batch_split(rollout_batch, split_count)
                rollout_batch = splits[split_index]
        else:
            # Colocated mode: generate locally
            rollout_batch = self._local_rollout(data)

        plosses, acces = self._forward(rollout_batch)

        # Weighted sum across TTT positions
        ploss_weight = [0.8**i for i in range(len(plosses))]
        ploss = (
            sum(ploss_weight[i] * plosses[i] for i in range(len(plosses)))
            / self.draft_accumulation_steps
        )
        ploss.backward()

        if global_step % self.draft_accumulation_steps == 0:
            self._optimizer.step()

        metrics = self._metrics_dict(
            plosses,
            acces,
            mode="train",
            should_log=(global_step % self._log_interval == 0),
        )
        return metrics

    def run_eval_step(self, rollout_batch_ref=None) -> dict:
        """
        Run one eval mini-batch.

        In colocated mode (rollout_batch_ref=None), iterates over the full
        eval DataLoader and generates rollout locally for each batch.

        In disaggregated mode, receives a single pre-computed RolloutBatch
        and evaluates it.  The pipeline calls this once per eval batch.
        """
        if self._eval_dataloader is None and rollout_batch_ref is None:
            return {}

        self._draft_model.eval()

        if rollout_batch_ref is not None:
            # Disaggregated: single batch from external rollout
            rollout_batch = ray.get(rollout_batch_ref)
            with torch.no_grad():
                plosses, acces = self._forward(rollout_batch)
            return self._metrics_dict(plosses, acces, mode="eval")

        # Colocated: iterate over full eval DataLoader
        eval_acces: List[List[torch.Tensor]] = [[] for _ in range(self.ttt_length)]
        eval_plosses: List[List[torch.Tensor]] = [[] for _ in range(self.ttt_length)]

        for data in self._eval_dataloader:
            with torch.no_grad():
                rollout_batch = self._local_rollout(data)
                plosses, acces = self._forward(rollout_batch)
            for i in range(len(acces)):
                eval_acces[i].append(acces[i])
                eval_plosses[i].append(plosses[i])

        eval_acces_mean = [torch.stack(a).mean() for a in eval_acces]
        eval_plosses_mean = [torch.stack(l).mean() for l in eval_plosses]
        return self._metrics_dict(eval_plosses_mean, eval_acces_mean, mode="eval")

    def save_checkpoint(self, epoch: int, step: int) -> str:
        """
        Save the draft model and training state to disk.
        All ranks participate (FSDP requires collective state_dict gather);
        only rank-0 writes to disk.
        Returns the checkpoint directory path (empty string on non-zero ranks).
        """
        epoch_output_dir = os.path.join(self.output_dir, f"epoch_{epoch}_step_{step}")
        if dist.get_rank(self._dp_group) == 0:
            os.makedirs(epoch_output_dir, exist_ok=True)
        dist.barrier(group=self._dp_group)

        with FSDP.state_dict_type(self._online_model, StateDictType.FULL_STATE_DICT):
            model_state_dict = self._online_model.state_dict()
            state_to_save = {
                "epoch": epoch,
                "global_step": step,
            }
            state_to_save.update(self._optimizer.state_dict())
            draft_state = {
                k.replace("draft_model.", ""): v
                for k, v in model_state_dict.items()
                if "draft_model." in k and "embed" not in k.lower()
            }

        if dist.get_rank(self._dp_group) == 0:
            torch.save(
                state_to_save,
                os.path.join(epoch_output_dir, "training_state.pt"),
            )
            self._online_model._fsdp_wrapped_module.draft_model.save_pretrained(
                epoch_output_dir, state_dict=draft_state
            )
        dist.barrier(group=self._dp_group)
        return epoch_output_dir if dist.get_rank(self._dp_group) == 0 else ""

    def shutdown(self) -> None:
        """Destroy distributed groups and release GPU memory."""
        try:
            from specforge.distributed import destroy_distributed

            destroy_distributed()
        except Exception:
            pass
        self._ready = False
