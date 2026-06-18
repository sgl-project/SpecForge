"""
RolloutWorker – Ray Actor that hosts the **target model** and serves
forward-pass results (hidden states + logits) to TrainWorkers.

One RolloutWorker occupies one GPU.  When rollout_tp_size > 1, multiple
RolloutWorkers form a TP group and process the same request together; only
the tp_rank-0 worker returns the resulting RolloutBatch.
"""

import logging
from typing import Optional

import ray
import torch

from specforge.ray_workers.worker_utils import RolloutBatch

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class RolloutWorker:
    """
    Hosts the target model (SGLang / HF / Custom backend) and generates
    Eagle3 training data (aux hidden-states + logits) for a given batch of
    token sequences.

    Lifecycle
    ---------
    1. Orchestrator creates N workers via ray.remote(...).options(...).remote(...)
    2. Orchestrator calls ``ray.get(worker.is_ready.remote())`` to confirm
       model loading finished.
    3. On each training step, orchestrator calls
       ``worker.generate_rollout_batch.remote(input_ids, attention_mask, loss_mask)``
       (Ray mode) or ``worker.generate_and_send.remote(input_ids, ..., dst_ranks)``
       (NCCL mode).
    4. At the end, orchestrator calls ``worker.shutdown.remote()``.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        tp_size: int,
        tp_rank: int,
        master_addr: str,
        master_port: int,
        target_model_path: str,
        backend: str,
        sglang_backend_kwargs: dict,
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        model_download_dir: Optional[str] = None,
        is_vlm: bool = False,
        timeout_minutes: int = 20,
        # NCCL transfer mode params
        transfer_backend: str = "ray",
        global_rank: Optional[int] = None,
        global_world_size: Optional[int] = None,
        global_master_addr: Optional[str] = None,
        global_master_port: Optional[int] = None,
        rollout_ranks: Optional[list] = None,
        # Method dispatch
        method: str = "eagle3",
        capture_layer_ids: Optional[list] = None,
    ) -> None:
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.is_vlm = is_vlm
        self._ready = False
        self._transfer_backend = transfer_backend
        self._global_rank = global_rank
        self._method = method

        # ── 1. Init distributed ───────────────────────────────────────────
        if transfer_backend == "nccl":
            from specforge.distributed_ray import (
                init_global_distributed,
                init_rollout_subgroup,
            )

            init_global_distributed(
                global_rank=global_rank,
                global_world_size=global_world_size,
                master_addr=global_master_addr,
                master_port=global_master_port,
                timeout_minutes=timeout_minutes,
            )
            init_rollout_subgroup(
                rollout_ranks=rollout_ranks,
                tp_size=tp_size,
            )
        else:
            from specforge.distributed_ray import init_rollout_distributed

            init_rollout_distributed(
                rank=rank,
                world_size=world_size,
                master_addr=master_addr,
                master_port=master_port,
                tp_size=tp_size,
                timeout_minutes=timeout_minutes,
            )

        # ── 2. Load target model ────────────────────────────────────────────
        dtype = getattr(torch, torch_dtype)

        if method == "eagle3":
            from specforge.modeling.target import get_eagle3_target_model

            self.target_model = get_eagle3_target_model(
                pretrained_model_name_or_path=target_model_path,
                backend=backend,
                torch_dtype=dtype,
                device="cuda",
                cache_dir=model_download_dir,
                trust_remote_code=trust_remote_code,
                **sglang_backend_kwargs,
            )
            # ── 3. Configure aux hidden-state layers ───────────────────────
            self.target_model.set_aux_hidden_states_layers(capture_layer_ids)
        elif method == "dflash":
            from specforge.modeling.target.dflash_target_model import (
                get_dflash_target_model,
            )

            self.target_model = get_dflash_target_model(
                pretrained_model_name_or_path=target_model_path,
                backend=backend,
                torch_dtype=dtype,
                device="cuda" if backend == "hf" else None,
                cache_dir=model_download_dir,
                trust_remote_code=trust_remote_code,
                **sglang_backend_kwargs,
            )
            # ── 3. Configure capture layers ────────────────────────────────
            if capture_layer_ids is not None:
                self.target_model.set_capture_layers(capture_layer_ids)

        self._ready = True
        logger.info(
            f"RolloutWorker rank={rank} tp_rank={tp_rank} "
            f"transfer={transfer_backend} ready."
        )

    def is_ready(self) -> bool:
        """Health-check: returns True once the model is fully loaded."""
        return self._ready

    def _run_forward(
        self,
        input_ids,
        attention_mask,
        loss_mask,
        pixel_values=None,
        image_grid_thw=None,
    ):
        """Run target model forward and return RolloutBatch (on GPU)."""
        device = torch.device("cuda")
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        loss_mask = loss_mask.to(device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)

        if self._method == "eagle3":
            output = self.target_model.generate_eagle3_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                is_vlm=self.is_vlm,
            )
            return RolloutBatch(
                input_ids=output.input_ids,
                attention_mask=output.attention_mask,
                loss_mask=output.loss_mask,
                hidden_states=output.hidden_states,
                target=output.target,
            )
        elif self._method == "dflash":
            output = self.target_model.generate_dflash_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
            )
            return RolloutBatch(
                input_ids=output.input_ids,
                attention_mask=output.attention_mask,
                loss_mask=output.loss_mask,
                hidden_states=output.hidden_states,
                target=None,
            )

    def generate_rollout_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw=None,
    ) -> Optional[RolloutBatch]:
        """
        Run forward and return RolloutBatch via Ray object store (CPU tensors).

        Used in Ray transfer mode (--transfer-backend ray).
        """
        batch = self._run_forward(
            input_ids, attention_mask, loss_mask, pixel_values, image_grid_thw
        )

        if self.tp_rank != 0:
            return None

        return RolloutBatch(
            input_ids=batch.input_ids.cpu(),
            attention_mask=batch.attention_mask.cpu(),
            loss_mask=batch.loss_mask.cpu(),
            hidden_states=batch.hidden_states.cpu(),
            target=batch.target.cpu() if batch.target is not None else None,
        )

    def generate_and_send(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        dst_ranks: list,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw=None,
    ) -> bool:
        """
        Run forward and send RolloutBatch directly to TrainWorkers via NCCL.

        Used in NCCL transfer mode (--transfer-backend nccl).
        Tensors stay on GPU — no CPU round-trip.

        Args:
            dst_ranks: List of global ranks to send to.

        Returns:
            True on success.
        """
        batch = self._run_forward(
            input_ids, attention_mask, loss_mask, pixel_values, image_grid_thw
        )

        if self.tp_rank != 0:
            return True

        from specforge.ray_workers.worker_utils import nccl_send_rollout_batch

        for dst in dst_ranks:
            nccl_send_rollout_batch(batch, dst_rank=dst)

        return True

    def get_model_config(self) -> dict:
        """Return the hf_config of the loaded model as a plain dict."""
        cfg = getattr(self.target_model, "hf_config", None)
        if cfg is None:
            inner = getattr(self.target_model, "model", None)
            cfg = getattr(inner, "config", None) if inner else None
        if cfg is None:
            return {}
        return cfg.to_dict() if hasattr(cfg, "to_dict") else {}

    def shutdown(self) -> None:
        """Destroy distributed process groups and free GPU memory."""
        try:
            from specforge.distributed import destroy_distributed

            destroy_distributed()
        except Exception:
            pass
        self._ready = False
