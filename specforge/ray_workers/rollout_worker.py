"""
RolloutWorker – Ray Actor that hosts the **target model** and serves
forward-pass results (hidden states + logits) to TrainWorkers.

One RolloutWorker occupies one GPU.  When rollout_tp_size > 1, multiple
RolloutWorkers form a TP group and process the same request together; only
the tp_rank-0 worker returns the resulting RolloutBatch.
"""

import logging
from typing import Dict, Optional

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
       ``worker.generate_rollout_batch.remote(input_ids, attention_mask, loss_mask)``.
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
        aux_hidden_states_layers,
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        model_download_dir: Optional[str] = None,
        is_vlm: bool = False,
        timeout_minutes: int = 20,
    ) -> None:
        """
        Parameters
        ----------
        rank:                    Global rank within the rollout process group.
        world_size:              Total rollout workers (== rollout_num_gpus).
        tp_size:                 Rollout TP degree.
        tp_rank:                 This worker's rank within its TP group.
        master_addr / port:      Rendezvous for torch.distributed.
        target_model_path:       HF model path or local directory.
        backend:                 "sglang" | "hf" | "custom".
        sglang_backend_kwargs:   Dict produced by SGLangBackendArgs.to_kwargs().
        aux_hidden_states_layers: Layer indices for Eagle3 aux hidden states
                                  (None → auto-detect from model config).
        torch_dtype:             "bfloat16" | "float16" | "float32".
        trust_remote_code:       Passed to model loaders.
        model_download_dir:      Optional local cache directory.
        is_vlm:                  Whether the target is a vision-language model.
        timeout_minutes:         NCCL init timeout.
        """
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.is_vlm = is_vlm
        self._ready = False

        # ── 1. Init distributed (rollout-side TP group) ────────────────────
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

        # ── 3. Configure aux hidden-state layers ───────────────────────────
        self.target_model.set_aux_hidden_states_layers(aux_hidden_states_layers)

        self._ready = True
        logger.info(f"RolloutWorker rank={rank} tp_rank={tp_rank} ready.")

    def is_ready(self) -> bool:
        """Health-check: returns True once the model is fully loaded."""
        return self._ready

    def generate_rollout_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw=None,
    ) -> Optional[RolloutBatch]:
        """
        Run a target-model forward pass and return the Eagle3 training data.

        All input tensors are expected on **CPU**; they are moved to CUDA
        inside this method.  The returned RolloutBatch has all tensors on
        CPU so it can travel through the Ray object store.

        Only tp_rank-0 returns a populated RolloutBatch; other ranks in the
        same TP group return None (they still participate in collective ops).

        Parameters
        ----------
        input_ids:      (B, seq_len) CPU tensor.
        attention_mask: (B, seq_len) CPU tensor.
        loss_mask:      (B, seq_len) CPU tensor.
        pixel_values:   Optional VLM pixel values.
        image_grid_thw: Optional VLM grid info.

        Returns
        -------
        RolloutBatch on tp_rank-0, None on other ranks.
        """
        device = torch.device("cuda")
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        loss_mask = loss_mask.to(device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)

        output = self.target_model.generate_eagle3_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            is_vlm=self.is_vlm,
        )

        if self.tp_rank != 0:
            return None

        return RolloutBatch(
            input_ids=output.input_ids.cpu(),
            attention_mask=output.attention_mask.cpu(),
            loss_mask=output.loss_mask.cpu(),
            hidden_states=output.hidden_states.cpu(),
            target=output.target.cpu(),
        )

    def get_model_config(self) -> dict:
        """Return the hf_config of the loaded model as a plain dict."""
        cfg = getattr(self.target_model, "hf_config", None)
        if cfg is None:
            # HF/Custom backend: config lives on the inner nn.Module
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
