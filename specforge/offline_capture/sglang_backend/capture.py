# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Version-pinned SGLang boundary for offline EAGLE3 data preparation."""

from __future__ import annotations

from array import array
from typing import List, Optional

import torch
import torch.distributed as dist
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_components.dp_attn import prepare_mlp_sync_batch_raw
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import require_mlp_sync, require_mlp_tp_gather

from specforge.distributed import get_tp_group

from .model_runner import SGLangRunner
from .utils import wrap_offline_eagle3_logits_processors


class OfflineSGLangCaptureBackend:
    """Frozen local target used only to materialize offline features."""

    def __init__(self, model_runner: SGLangRunner) -> None:
        self.model_runner = model_runner

    @classmethod
    def build(
        cls,
        pretrained_model_name_or_path: str,
        *,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "OfflineSGLangCaptureBackend":
        tp_size = dist.get_world_size(get_tp_group())
        server_args = ServerArgs(
            model_path=pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype if torch_dtype is not None else "auto",
            enable_return_hidden_states=True,
            disable_cuda_graph=True,
            chunked_prefill_size=-1,
            tp_size=tp_size,
            pp_size=1,
            **kwargs,
        )

        tp_rank = dist.get_rank(get_tp_group())
        moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
        model_config = ModelConfig.from_server_args(server_args)
        model_runner = SGLangRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=torch.cuda.current_device(),
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=server_args.ep_size,
            pp_rank=0,
            pp_size=1,
            server_args=server_args,
            nccl_port=None,
            is_draft_worker=False,
        )
        model_runner.alloc_memory_pool()
        model_runner.init_attention_backends()
        model_runner.init_cuda_graphs()
        wrap_offline_eagle3_logits_processors(model_runner.model)
        return cls(model_runner)

    def set_eagle3_capture_layers(self, layer_ids: Optional[List[int]] = None) -> None:
        self.model_runner.model.set_eagle3_layers_to_capture(layer_ids)

    def set_capture_layers(
        self,
        layer_ids: Optional[List[int]] = None,
        *,
        capture_method: str,
    ) -> None:
        """Set auxiliary layers through the strategy's SGLang capture API."""

        setter_name = {
            "eagle3": "set_eagle3_layers_to_capture",
            "dflash": "set_dflash_layers_to_capture",
        }.get(capture_method)
        if setter_name is None:
            raise ValueError(
                "offline SGLang capture method must be 'eagle3' or 'dflash', "
                f"got {capture_method!r}"
            )
        setter = getattr(self.model_runner.model, setter_name, None)
        if not callable(setter):
            raise RuntimeError(
                f"target model does not expose SGLang capture hook {setter_name!r}"
            )
        setter(layer_ids)

    def _maybe_prepare_mlp_sync_batch(self, batch: ScheduleBatch) -> None:
        if require_mlp_sync(self.model_runner.server_args):
            prepare_mlp_sync_batch_raw(
                batch,
                dp_size=self.model_runner.server_args.dp_size,
                attn_tp_size=1,
                attn_cp_size=getattr(self.model_runner.server_args, "attn_cp_size", 1),
                tp_group=self.model_runner.tp_group,
                get_idle_batch=None,
                disable_cuda_graph=self.model_runner.server_args.disable_cuda_graph,
                require_mlp_tp_gather=require_mlp_tp_gather(
                    self.model_runner.server_args
                ),
                disable_overlap_schedule=self.model_runner.server_args.disable_overlap_schedule,
                offload_tags=set(),
            )

    @torch.no_grad()
    def _forward_extend(self, reqs: list[Req]):
        cache_params = CacheInitParams(
            disable=False,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            page_size=self.model_runner.server_args.page_size,
        )
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=RadixCache(cache_params),
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        self._maybe_prepare_mlp_sync_batch(batch)
        if getattr(batch, "prefill_input_ids_cpu", None) is not None:
            batch.input_ids = batch.prefill_input_ids_cpu.to(
                batch.device, non_blocking=True
            )
            batch.prefill_input_ids_cpu = None
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        forward_batch = ForwardBatch.init_new(batch, self.model_runner)
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        output = self.model_runner.forward(forward_batch)
        return output.logits_output if hasattr(output, "logits_output") else output

    def _clear_pools(self) -> None:
        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()

    @torch.no_grad()
    def capture_eagle3(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ):
        """Capture per-request auxiliary and final hidden states without logits."""

        sampling_params = SamplingParams(temperature=0, max_new_tokens=1, top_k=1)
        reqs: list[Req] = []
        data = []
        input_rows = torch.split(input_ids, 1, dim=0)
        attention_rows = torch.split(attention_mask, 1, dim=0)
        loss_rows = torch.split(loss_mask, 1, dim=0)

        for idx, (input_row, attention_row, loss_row) in enumerate(
            zip(input_rows, attention_rows, loss_rows)
        ):
            req = Req(
                rid=str(idx),
                origin_input_text="",
                origin_input_ids=input_row.view(-1).tolist(),
                sampling_params=sampling_params,
            )
            req.full_untruncated_fill_ids = array("q", req.origin_input_ids)
            req.fill_len = len(req.full_untruncated_fill_ids)
            req.extend_input_len = req.fill_len - len(req.prefix_indices)
            req.logprob_start_len = len(req.origin_input_ids) - 1
            reqs.append(req)
            data.append((input_row, attention_row, loss_row))

        input_lens = [len(req.origin_input_ids) for req in reqs]
        try:
            output = self._forward_extend(reqs)
            aux_hidden_states = getattr(output, "aux_hidden_states", None)
            last_hidden_states = getattr(output, "last_hidden_states", None)
            if aux_hidden_states is None or last_hidden_states is None:
                raise RuntimeError(
                    "SGLang did not return the hidden states required for "
                    "offline feature preparation"
                )
            aux_rows = torch.split(aux_hidden_states, input_lens, dim=0)
            last_rows = torch.split(last_hidden_states, input_lens, dim=0)
        finally:
            self._clear_pools()

        return data, aux_rows, last_rows

    def capture(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ):
        """Capture generic auxiliary and final target states."""

        return self.capture_eagle3(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )


__all__ = ["OfflineSGLangCaptureBackend"]
