"""
TrainWorker – Ray Actor that hosts the **draft model** and drives
the Eagle3 forward/backward training step.

One TrainWorker occupies one GPU.  Multiple TrainWorkers form a torch.distributed
process group and use FSDP for data-parallel gradient synchronisation.

Colocated vs. Disaggregated
----------------------------
* colocated   (rollout_worker_ref is None):
    The TrainWorker also loads the target model on the same GPU and runs
    rollout internally before the draft-model forward pass.
* disaggregated (rollout_worker_ref is an actor handle):
    The TrainWorker sends the raw batch to its paired RolloutWorker via
    Ray and waits for the RolloutBatch before proceeding.
"""

import hashlib
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

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
        # optional colocated rollout
        rollout_worker_ref=None,           # ray.ObjectRef to RolloutWorker or None
        target_model_backend: str = "sglang",
        sglang_backend_kwargs: Optional[dict] = None,
        aux_hidden_states_layers=None,
        # dataset / dataloader
        train_data_path: Optional[str] = None,
        eval_data_path: Optional[str] = None,
        chat_template: str = "qwen",
        max_length: int = 4096,
        batch_size: int = 1,              # per-DP-rank batch size
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
        self.output_dir = output_dir
        self.is_vlm = is_vlm
        self.batch_size = batch_size
        self.target_batch_size = tp_size * batch_size
        self.max_length = max_length
        self._ready = False
        self._rollout_worker_ref = rollout_worker_ref

        # adjust accumulation for SP (mirrors sp_sanity_check in train_eagle3.py)
        if self.sp_size > 1:
            self.draft_accumulation_steps *= self.sp_size

        # ── 1. Init distributed ────────────────────────────────────────────
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
        from specforge.distributed import get_dp_group, get_draft_dp_group, get_tp_group

        self._tp_group = get_tp_group()
        self._dp_group = get_dp_group()
        self._tp_rank = dist.get_rank(self._tp_group)

        # ── 2. Load draft model ────────────────────────────────────────────
        from specforge.modeling import AutoDraftModelConfig, AutoEagle3DraftModel
        from specforge.utils import create_draft_config_from_target, get_last_checkpoint

        ckpt_info = (0, 0)
        resume_state = None

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

        draft_model_last_checkpoint = None
        is_resume_checkpoint = False

        if ckpt_dir is not None and os.path.isdir(ckpt_dir):
            self._draft_model_config = AutoDraftModelConfig.from_file(
                os.path.join(ckpt_dir, "config.json")
            )
            draft_model_last_checkpoint = ckpt_dir

        if resume and os.path.isdir(output_dir):
            draft_model_last_checkpoint, ckpt_info = get_last_checkpoint(output_dir)
            is_resume_checkpoint = True

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

        if is_resume_checkpoint and draft_model_last_checkpoint:
            training_state_path = os.path.join(
                draft_model_last_checkpoint, "training_state.pt"
            )
            if os.path.exists(training_state_path):
                resume_state = torch.load(
                    training_state_path, map_location="cpu", weights_only=False
                )

        self._draft_model.load_embedding(
            target_model_path, embedding_key=embedding_key
        )
        self._draft_model.freeze_embedding()

        # ── 3. Colocated: also load target model ──────────────────────────
        self._target_model = None
        if rollout_worker_ref is None:
            # Colocated mode: load target model directly on this GPU
            dtype = getattr(torch, torch_dtype)
            from specforge.args import SGLangBackendArgs
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
            self._target_model.set_aux_hidden_states_layers(aux_hidden_states_layers)

        # ── 4. Build OnlineEagle3Model + FSDP ─────────────────────────────
        from specforge.core import OnlineEagle3Model

        eagle3_model = OnlineEagle3Model(
            draft_model=self._draft_model,
            length=ttt_length,
            attention_backend=attention_backend,
        )
        self._eagle3_model = FSDP(
            eagle3_model,
            use_orig_params=True,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            process_group=dist.group.WORLD,
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

        # ── 6. Build dataset & DataLoader ─────────────────────────────────
        self._train_dataloader = None
        self._eval_dataloader = None
        self._train_iter = None
        self._train_data_path = train_data_path
        self._eval_data_path = eval_data_path

        if train_data_path is not None:
            self._build_dataloaders(
                train_data_path=train_data_path,
                eval_data_path=eval_data_path,
                chat_template=chat_template,
                max_length=max_length,
                cache_dir=cache_dir,
                dataloader_num_workers=dataloader_num_workers,
                build_dataset_num_proc=build_dataset_num_proc,
                is_preformatted=is_preformatted,
                train_only_last_turn=train_only_last_turn,
                target_model_path=target_model_path,
                trust_remote_code=trust_remote_code,
            )

        self._ready = True
        logger.info(f"TrainWorker rank={rank} ready.")

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _build_dataloaders(
        self,
        train_data_path: str,
        eval_data_path: Optional[str],
        chat_template: str,
        max_length: int,
        cache_dir: str,
        dataloader_num_workers: int,
        build_dataset_num_proc: int,
        is_preformatted: bool,
        train_only_last_turn: bool,
        target_model_path: str,
        trust_remote_code: bool,
    ) -> None:
        from transformers import AutoTokenizer

        from datasets import Dataset
        from specforge.data import (
            build_eagle3_dataset,
            generate_vocab_mapping_file,
            prepare_dp_dataloaders,
        )
        from specforge.distributed import get_dp_group, get_draft_dp_group
        from specforge.utils import rank_0_priority, safe_conversations_generator

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

        with rank_0_priority():
            train_eagle3_dataset = build_eagle3_dataset(
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
            vocab_mapping_path = generate_vocab_mapping_file(
                dataset=train_eagle3_dataset,
                target_vocab_size=self._draft_model_config.vocab_size,
                draft_vocab_size=self._draft_model_config.draft_vocab_size,
                cache_dir=os.path.join(cache_dir, "vocab_mapping"),
                cache_key=cache_key,
            )

        self._draft_model.load_vocab_mapping(vocab_mapping_path)

        # Use draft_dp_group as sampler group for SP, dp_group otherwise
        sampler_group = (
            get_draft_dp_group()
            if self.sp_size > 1
            else get_dp_group()
        )

        self._train_dataloader = prepare_dp_dataloaders(
            train_eagle3_dataset,
            self.target_batch_size,
            num_workers=dataloader_num_workers,
            shuffle=True,
            process_group=sampler_group,
            is_vlm=self.is_vlm,
        )

        if eval_data_path is not None:
            eval_dataset_raw = Dataset.from_generator(
                generator=safe_conversations_generator,
                gen_kwargs={"file_path": eval_data_path},
            )
            eval_eagle3_dataset = build_eagle3_dataset(
                eval_dataset_raw,
                tokenizer,
                chat_template,
                max_length,
                is_vlm=self.is_vlm,
                num_proc=build_dataset_num_proc,
                is_preformatted=is_preformatted,
                train_only_last_turn=train_only_last_turn,
            )
            self._eval_dataloader = prepare_dp_dataloaders(
                eval_eagle3_dataset,
                self.target_batch_size,
                num_workers=dataloader_num_workers,
                shuffle=False,
                process_group=sampler_group,
                is_vlm=self.is_vlm,
            )

    def _do_rollout(self, data: dict) -> "RolloutBatch":
        """
        Obtain Eagle3 training data for *data* either from the locally loaded
        target model (colocated) or by calling the paired RolloutWorker
        (disaggregated).
        """
        from specforge.ray_workers.worker_utils import RolloutBatch, batch_to_device

        if self._target_model is not None:
            # ── Colocated mode ─────────────────────────────────────────────
            output = self._target_model.generate_eagle3_data(
                input_ids=data["input_ids"].cuda(),
                attention_mask=data["attention_mask"].cuda(),
                loss_mask=data["loss_mask"].cuda(),
            )
            return RolloutBatch(
                input_ids=output.input_ids.cpu(),
                attention_mask=output.attention_mask.cpu(),
                loss_mask=output.loss_mask.cpu(),
                hidden_states=output.hidden_states.cpu(),
                target=output.target.cpu(),
            )
        else:
            # ── Disaggregated mode ─────────────────────────────────────────
            result = ray.get(
                self._rollout_worker_ref.generate_rollout_batch.remote(
                    data["input_ids"],
                    data["attention_mask"],
                    data["loss_mask"],
                )
            )
            return result

    def _forward(self, rollout_batch: "RolloutBatch"):
        """
        Run OnlineEagle3Model.forward() with pre-computed hidden states and
        target logits (no internal target-model call).
        """
        from specforge.ray_workers.worker_utils import batch_shard_by_tp, batch_to_device

        device = torch.device("cuda")
        rb = batch_to_device(rollout_batch, device)

        # TP shard: each worker takes its slice of the batch dimension
        rb = batch_shard_by_tp(rb, self.tp_size, self._tp_rank)

        plosses, _, acces = self._eagle3_model(
            input_ids=rb.input_ids,
            attention_mask=rb.attention_mask,
            loss_mask=rb.loss_mask,
            target=rb.target,
            hidden_states=rb.hidden_states,
            position_ids=rb.position_ids,
        )
        return plosses, acces

    def _metrics_dict(
        self,
        plosses: List[torch.Tensor],
        acces: List[torch.Tensor],
        mode: str = "train",
    ) -> dict:
        """Reduce metrics across DP and return a plain Python dict."""
        accuracies = torch.stack(acces)
        losses = torch.stack(plosses)
        dist.all_reduce(accuracies, op=dist.ReduceOp.AVG)
        dist.all_reduce(losses, op=dist.ReduceOp.AVG)
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
        self._train_iter = iter(self._train_dataloader) if self._train_dataloader else None

    def run_step(self, global_step: int, skip: bool = False) -> Optional[dict]:
        """
        Execute one training step:
          1. Fetch next batch from local DataLoader.
          2. Run rollout (colocated or disaggregated).
          3. Forward / backward / optimizer update.
          4. Return metrics dict (only rank-0 returns non-None; others return
             the same dict to avoid blocking, but orchestrator uses rank-0's).

        Args:
            global_step: Current global training step (for accumulation logic).
            skip:        If True, advance the iterator without training
                         (used when resuming mid-epoch).
        Returns:
            Metrics dict on every rank (orchestrator takes rank-0's copy).
        """
        if self._train_iter is None:
            raise RuntimeError("Call set_epoch() before run_step().")

        data = next(self._train_iter)

        if skip:
            return None

        self._draft_model.train()

        rollout_batch = self._do_rollout(data)
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

        metrics = self._metrics_dict(plosses, acces, mode="train")
        return metrics

    def run_eval_step(self) -> dict:
        """Run one eval mini-batch and return metrics (no backward)."""
        if self._eval_dataloader is None:
            return {}
        self._draft_model.eval()
        eval_acces: List[List[torch.Tensor]] = [[] for _ in range(self.ttt_length)]
        eval_plosses: List[List[torch.Tensor]] = [[] for _ in range(self.ttt_length)]

        for data in self._eval_dataloader:
            with torch.no_grad():
                rollout_batch = self._do_rollout(data)
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
        if dist.get_rank() == 0:
            os.makedirs(epoch_output_dir, exist_ok=True)
        dist.barrier()

        with FSDP.state_dict_type(self._eagle3_model, StateDictType.FULL_STATE_DICT):
            model_state_dict = self._eagle3_model.state_dict()
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

        if dist.get_rank() == 0:
            torch.save(
                state_to_save,
                os.path.join(epoch_output_dir, "training_state.pt"),
            )
            self._eagle3_model._fsdp_wrapped_module.draft_model.save_pretrained(
                epoch_output_dir, state_dict=draft_state
            )
        dist.barrier()
        return epoch_output_dir if dist.get_rank() == 0 else ""

    def shutdown(self) -> None:
        """Destroy distributed groups and release GPU memory."""
        try:
            from specforge.distributed import destroy_distributed

            destroy_distributed()
        except Exception:
            pass
        self._ready = False
