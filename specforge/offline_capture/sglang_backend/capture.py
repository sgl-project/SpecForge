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

import logging
import socket
from array import array
from typing import List, Optional

import torch
import torch.distributed as dist
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import bootstrap
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_components.dp_attn import prepare_mlp_sync_batch_raw
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.runtime_context import (
    SpawnRanks,
    get_device,
    get_schedule,
    publish,
    spawn_world_rank,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import require_mlp_sync, require_mlp_tp_gather

from .model_runner import SGLangRunner
from .utils import wrap_offline_eagle3_logits_processors

logger = logging.getLogger(__name__)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# SGLang capture hooks tried in order for each capture method. K3-class targets
# expose a native DSpark hook; dense targets serve the same auxiliary
# hidden-state layout through the DFlash hook, so DSpark falls back to it.
_CAPTURE_LAYER_HOOKS = {
    "eagle3": ("set_eagle3_layers_to_capture",),
    "dflash": ("set_dflash_layers_to_capture",),
    "dspark": ("set_dspark_layers_to_capture", "set_dflash_layers_to_capture"),
    "hspec": ("set_dflash_layers_to_capture",),
}


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
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
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

        gpu_id = torch.get_device_module().current_device()
        # Offline capture bypasses the scheduler, so publish the runner's
        # config and build the parallel runtime here, mirroring the upstream
        # one-batch benchmark entry point.
        publish(
            server_args,
            role="scheduler",
            ranks=SpawnRanks(
                world_rank=spawn_world_rank(server_args, tp_rank=0, pp_rank=0),
                gpu_id=gpu_id,
            ),
        )
        model_config = ModelConfig.from_server_args(server_args)
        nccl_port = _find_free_port()
        bootstrap.init_parallel_runtime(
            server_args=server_args,
            device=get_device().device,
            dist_port=nccl_port,
        )
        bootstrap.init_layer_runtime(model_config=model_config)
        model_runner = SGLangRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            nccl_port=nccl_port,
            server_args=server_args,
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

        setter_names = _CAPTURE_LAYER_HOOKS.get(capture_method)
        if setter_names is None:
            raise ValueError(
                "offline SGLang capture method must be 'eagle3', 'dflash', "
                "'dspark', or 'hspec', "
                f"got {capture_method!r}"
            )
        for setter_name in setter_names:
            setter = getattr(self.model_runner.model, setter_name, None)
            if not callable(setter):
                continue
            if setter_name != setter_names[0]:
                logger.info(
                    "capture method %r resolved through compatibility hook %s",
                    capture_method,
                    setter_name,
                )
            setter(layer_ids)
            return
        raise RuntimeError(
            "target model does not expose a compatible SGLang capture hook; "
            f"tried {setter_names!r}"
        )

    def _maybe_prepare_mlp_sync_batch(self, batch: ScheduleBatch) -> None:
        if require_mlp_sync():
            prepare_mlp_sync_batch_raw(
                batch,
                model_runner=self.model_runner,
                get_idle_batch=None,
                disable_cuda_graph=self.model_runner.server_args.disable_cuda_graph,
                require_mlp_tp_gather=require_mlp_tp_gather(),
                disable_overlap_schedule=get_schedule().disable_overlap_schedule,
                offload_tags=set(),
            )

    @torch.no_grad()
    def capture_hspec(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ):
        """Capture H-Spec hidden states through the SGLang dflash hook.

        SGLang's generic offline boundary currently captures only hidden
        states; selected post-RoPE K/V require a separate target-side
        capture artifact.  Return explicit placeholders for the KV fields so
        the local capture path fails loudly if a trainer consumes them before
        the fused SGLang producer is available.
        """

        data = list(
            zip(
                torch.split(input_ids, 1, dim=0),
                torch.split(attention_mask, 1, dim=0),
                torch.split(loss_mask, 1, dim=0),
            )
        )
        rows = [input_row.view(-1).tolist() for input_row, _, _ in data]
        input_lens = [len(row) for row in rows]

        sampling_params = SamplingParams(temperature=0, max_new_tokens=1, top_k=1)
        reqs: list[Req] = []
        for idx, input_row in enumerate(rows):
            req = Req(
                rid=str(idx),
                origin_input_text="",
                origin_input_ids=list(input_row),
                sampling_params=sampling_params,
            )
            req.full_untruncated_fill_ids = array("q", req.origin_input_ids)
            req.set_extend_range(
                len(req.prefix_indices), len(req.full_untruncated_fill_ids)
            )
            req.logprob_start_len = len(req.origin_input_ids) - 1
            reqs.append(req)

        try:
            output = self._forward_extend(reqs)
            forward_batch = getattr(output, "forward_batch", None)
            if forward_batch is None:
                forward_batch = self.model_runner._last_forward_batch
            aux_hidden_states = getattr(output, "aux_hidden_states", None)
            last_hidden_states = getattr(output, "last_hidden_states", None)
            if aux_hidden_states is None or last_hidden_states is None:
                raise RuntimeError(
                    "SGLang did not return hidden states required for H-Spec capture"
                )
            kv_layer_ids = self._hspec_kv_layer_ids()
            kv_rows = self._capture_target_kv(
                forward_batch, input_lens, kv_layer_ids
            )
            hidden_rows = torch.split(aux_hidden_states, input_lens, dim=0)
            last_rows = torch.split(last_hidden_states, input_lens, dim=0)
            features = []
            for hidden_row, last_row, kv_row, (input_row, mask_row, loss_row) in zip(
                hidden_rows,
                last_rows,
                kv_rows,
                data,
            ):
                keys, values = zip(*kv_row)
                # Flatten each layer to [seq, heads * head_dim] first, then
                # concatenate layers on the last axis so the trainer can slice
                # per-layer widths from [seq, layers * heads * head_dim].
                flat_keys = [key.reshape(key.shape[0], -1) for key in keys]
                flat_values = [value.reshape(value.shape[0], -1) for value in values]
                features.append(
                    {
                        "input_ids": input_row,
                        "loss_mask": loss_row,
                        # 1 = real context token, 0 = padding; the trainer
                        # masks padded target K/V out of the prefix softmax.
                        "prefix_masks": mask_row,
                        "hidden_states": hidden_row,
                        "target_last_hidden_states": last_row,
                        "selected_target_k": torch.cat(flat_keys, dim=-1),
                        "selected_target_v": torch.cat(flat_values, dim=-1),
                    }
                )
            return features
        finally:
            self._clear_pools()

    def set_hspec_kv_layer_ids(self, layer_ids) -> None:
        self._hspec_kv_layer_ids_override = [int(x) for x in layer_ids]

    def _hspec_kv_layer_ids(self) -> list[int]:
        override = getattr(self, "_hspec_kv_layer_ids_override", None)
        if override:
            return [int(x) for x in override]
        config = self.model_runner.model_config
        hf_config = getattr(config, "hf_config", config)
        method = getattr(hf_config, "hspec_config", None)
        if method is None:
            from sglang.srt.configs.model_config import ModelConfig

            draft_config = ModelConfig.from_server_args(
                self.model_runner.server_args,
                model_path=(
                    self.model_runner.server_args.speculative_draft_model_path
                ),
                is_draft_model=True,
            )
            method = getattr(draft_config.hf_config, "hspec_config", {}) or {}
        layer_ids = method.get("attn_kv_layer_ids")
        if layer_ids is None:
            layer_ids = method.get("target_kv_layer_ids")
        if not layer_ids:
            raise ValueError(
                "target config must define hspec_config.attn_kv_layer_ids"
            )
        return [int(layer_id) for layer_id in layer_ids]

    def _capture_target_kv(
        self,
        forward_batch,
        input_lens: list[int],
        kv_layer_ids: list[int],
    ):
        """Read committed post-RoPE K/V from the target paged KV pool.

        Output layout is ``[seq, kv_heads, head_dim]`` per selected layer.
        Quantized KV is rejected instead of silently dequantizing raw cache.
        """
        pool = self.model_runner.token_to_kv_pool
        if bool(getattr(pool, "is_quantized_kv_cache", False)):
            raise RuntimeError(
                "H-Spec offline capture does not support quantized KV cache"
            )
        out_cache_loc = forward_batch.out_cache_loc
        if out_cache_loc is None or int(out_cache_loc.numel()) != sum(input_lens):
            raise RuntimeError(
                "forward batch out_cache_loc does not match captured token interval"
            )
        per_request = []
        offset = 0
        for length in input_lens:
            loc = out_cache_loc[offset : offset + length]
            offset += length
            layer_kv = []
            for layer_id in kv_layer_ids:
                key_cache, value_cache = pool.get_kv_buffer(layer_id)
                if key_cache.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    raise RuntimeError(
                        f"unsupported KV dtype {key_cache.dtype} for H-Spec capture"
                    )
                if key_cache.dim() == 4:
                    page_size = key_cache.shape[1]
                    key = key_cache[loc // page_size, loc % page_size]
                    value = value_cache[loc // page_size, loc % page_size]
                elif key_cache.dim() == 3:
                    key = key_cache[loc]
                    value = value_cache[loc]
                else:
                    raise RuntimeError(
                        "unsupported target KV cache layout: "
                        f"K{tuple(key_cache.shape)}"
                    )
                key = key.contiguous()
                value = value.contiguous()
                if key.device.type == "npu":
                    # NDVI/NZ layout for the NPU attention kernels; a no-op
                    # elsewhere where the dense layout is already canonical.
                    key = torch.ops.npu.npu_format_cast(key, 2)
                    value = torch.ops.npu.npu_format_cast(value, 2)
                layer_kv.append((key, value))
            per_request.append(layer_kv)
        return per_request

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
        forward_batch = ForwardBatch.init_new(
            batch,
            self.model_runner,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            return_hidden_states_before_norm=False,
        )
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        self.model_runner._last_forward_batch = forward_batch
        output = self.model_runner.forward(forward_batch)
        return output.logits_output if hasattr(output, "logits_output") else output

    def _clear_pools(self) -> None:
        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()

    @torch.no_grad()
    def capture_rows(self, input_ids: list[list[int]]):
        """Capture variable-length request rows in one packed prefill.

        Returns ``(aux_rows, last_rows)``: per-row auxiliary and final hidden
        states split from the packed forward, so callers never build padded
        tensors. The KV/request pools are cleared after every call.
        """

        if not input_ids:
            return (), ()
        if any(not row for row in input_ids):
            raise ValueError("SGLang capture rows must contain at least one token")
        sampling_params = SamplingParams(temperature=0, max_new_tokens=1, top_k=1)
        reqs: list[Req] = []
        for idx, input_row in enumerate(input_ids):
            req = Req(
                rid=str(idx),
                origin_input_text="",
                origin_input_ids=list(input_row),
                sampling_params=sampling_params,
            )
            req.full_untruncated_fill_ids = array("q", req.origin_input_ids)
            req.set_extend_range(
                len(req.prefix_indices), len(req.full_untruncated_fill_ids)
            )
            req.logprob_start_len = len(req.origin_input_ids) - 1
            reqs.append(req)

        input_lens = [len(req.origin_input_ids) for req in reqs]
        try:
            output = self._forward_extend(reqs)
            aux_hidden_states = getattr(output, "aux_hidden_states", None)
            last_hidden_states = getattr(output, "last_hidden_states", None)
            if aux_hidden_states is None or last_hidden_states is None:
                raise RuntimeError(
                    "SGLang did not return the hidden states required for capture"
                )
            aux_rows = torch.split(aux_hidden_states, input_lens, dim=0)
            last_rows = torch.split(last_hidden_states, input_lens, dim=0)
        finally:
            self._clear_pools()
        return aux_rows, last_rows

    @torch.no_grad()
    def capture_eagle3(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ):
        """Capture per-request auxiliary and final hidden states without logits.

        Padded rows are captured as given (padding positions included), which
        keeps offline feature preparation byte-identical.
        """

        data = list(
            zip(
                torch.split(input_ids, 1, dim=0),
                torch.split(attention_mask, 1, dim=0),
                torch.split(loss_mask, 1, dim=0),
            )
        )
        aux_rows, last_rows = self.capture_rows(
            [input_row.view(-1).tolist() for input_row, _, _ in data]
        )
        return data, aux_rows, last_rows

    def capture(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ):
        """Capture generic auxiliary and final target states."""

        raise NotImplementedError(
            "OfflineSGLangCaptureBackend.capture is not used; "
            "OfflineSGLangCapture dispatches by capture method"
        )


__all__ = ["OfflineSGLangCaptureBackend"]
