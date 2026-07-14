# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""SGLangCaptureBackend: the ONE sglang-version-pinned capture boundary (Phase B2).

Before this module, both ``SGLangEagle3TargetEngine`` and
``SGLangDFlashTargetEngine`` imported ~20 sglang internals directly and each
carried its own near-duplicate ``_extend`` forward glue — the two copies had even
drifted to *different* sglang API versions (the eagle3 copy used the current
module-level ``prepare_mlp_sync_batch_raw(..., attn_cp_size=)``; the dflash copy
used the older ``Scheduler.prepare_mlp_sync_batch_raw(..., spec_algorithm=)``,
which no longer exists on sglang 0.5.9). That is the "coupled and tangled" state:
a sglang bump touched every ``*TargetEngine`` subclass, and the two copies could
silently diverge.

This backend owns **every** sglang symbol and the single extend/capture forward
primitive. The algorithm engines (``Eagle3TargetEngine`` / ``DFlashTargetEngine``
sglang backends) now *compose* one of these and shape its raw per-request output
in pure torch — they import zero sglang internals. So:

* a sglang version bump touches only this module (+ ``model_runner.py`` / ``utils.py``);
* the extend forward + the (version-pinned) mlp-sync live in exactly one place
  (``_forward_extend`` / ``_maybe_prepare_mlp_sync_batch``), so eagle3 and dflash
  can no longer drift;
* ``import specforge`` still works without the pinned sglang, because this module
  is imported lazily inside each engine's ``from_pretrained`` (never at package load).

This module is intentionally the *only* place that ``import``s sglang internals
for target capture; keep it that way.
"""

from __future__ import annotations

from array import array
from typing import List, Optional

import torch
import torch.distributed as dist
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

# prepare_mlp_sync_batch_raw is a module-level function, not a Scheduler method.
# sglang 0.5.14 moved it from managers.scheduler_dp_attn_mixin to
# managers.scheduler_components.dp_attn.
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
from .utils import LogitsProcessorForEAGLE3, wrap_eagle3_logits_processors_in_module


def _get_sharded_return(
    input_: torch.Tensor,
    input_lens: list[int],
    valid_input_lens: list[int],
    valid_indices: list[int],
) -> list[Optional[torch.Tensor]]:
    out: list[Optional[torch.Tensor]] = [None] * len(input_lens)
    input_scatter = torch.split(input_, valid_input_lens, dim=0)
    for j, idx in enumerate(valid_indices):
        out[idx] = input_scatter[j]
    return out


class SGLangCaptureBackend:
    """A frozen target's SGLang ``ModelRunner`` + the capture-forward primitives.

    Constructed once via :meth:`build` (which owns the ``ServerArgs`` /
    ``ModelConfig`` / ``SGLangRunner`` wiring), then reused by a single target
    engine. All sglang-internal state lives here.
    """

    def __init__(self, model_runner: SGLangRunner):
        self.model_runner = model_runner

    @classmethod
    def build(
        cls,
        pretrained_model_name_or_path: str,
        *,
        torch_dtype: torch.dtype = None,
        trust_remote_code: bool = False,
        wrap_eagle3_logits: bool = False,
        return_full_logits: bool = False,
        **kwargs,
    ) -> "SGLangCaptureBackend":
        """Construct the sglang ModelRunner for a frozen target.

        The construction is identical for every algorithm; ``wrap_eagle3_logits``
        is the only per-algorithm knob (eagle3 wraps the logits processors to
        capture aux hidden states, dflash does not). ``is_draft_worker=False`` is
        passed explicitly (the ModelRunner default, kept explicit as in the
        original eagle3 path).
        """
        tp_size = dist.get_world_size(get_tp_group())
        # NOTE: sglang 0.5.13+ requires dtype to be non-None. If torch_dtype is None,
        # use "auto" to let sglang decide the dtype.
        dtype_arg = torch_dtype if torch_dtype is not None else "auto"
        # NOTE: DFlash/EAGLE3 prefill the whole batch in one _forward_extend call
        # (no scheduler chunking). sglang 0.5.14's eager runner pre-allocates
        # per-call buffers sized to chunked_prefill_size (default 8192), and copies
        # into them fail with a shape mismatch when our token count exceeds that.
        # Disable chunked prefill so the ceiling grows to max_total_num_tokens.
        server_args = ServerArgs(
            model_path=pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            dtype=dtype_arg,
            enable_return_hidden_states=True,
            disable_cuda_graph=True,  # we use piecewise cuda graph for prefill instead
            chunked_prefill_size=-1,
            tp_size=tp_size,
            pp_size=1,
            **kwargs,
        )

        tp_rank = dist.get_rank(get_tp_group())
        moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
        model_config = ModelConfig.from_server_args(server_args)
        # - Added is_draft_worker=False parameter (new in 0.5.9)
        # - Other new parameters (dp_rank, attn_cp_rank, moe_dp_rank, etc.) use defaults
        model_runner = SGLangRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=torch.cuda.current_device(),
            tp_rank=dist.get_rank(get_tp_group()),
            tp_size=server_args.tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=server_args.ep_size,
            pp_rank=0,
            pp_size=1,
            server_args=server_args,
            nccl_port=None,
            is_draft_worker=False,
        )
        # sglang 0.5.14 split the post-load setup out of ModelRunner.initialize()
        # (which now only loads the weights). The scheduler/TpModelWorker perform
        # these steps explicitly; since we drive the ModelRunner directly, we must
        # replicate them so `req_to_token_pool`/`token_to_kv_pool_allocator` exist
        # and forward() has an attention backend and (eager) runner.
        model_runner.alloc_memory_pool()
        model_runner.init_attention_backends()
        model_runner.init_cuda_graphs()
        if wrap_eagle3_logits:
            wrap_eagle3_logits_processors_in_module(
                model_runner.model, return_full_logits=return_full_logits
            )
        return cls(model_runner)

    # --- capture-layer selection -------------------------------------------

    def set_eagle3_capture_layers(
        self, layer_ids: Optional[List[int]] = None, *, if_supported: bool = False
    ) -> None:
        """Tell the model which layers to capture aux hidden states from.

        ``if_supported`` guards on ``hasattr`` (dflash calls it that way — some
        target models lack the method); eagle3 calls it unguarded so a missing
        method surfaces loudly, exactly as before.
        """
        model = self.model_runner.model
        if if_supported and not hasattr(model, "set_eagle3_layers_to_capture"):
            return
        model.set_eagle3_layers_to_capture(layer_ids)

    # --- shared forward primitive ------------------------------------------

    def _maybe_prepare_mlp_sync_batch(self, batch: ScheduleBatch):
        if require_mlp_sync(self.model_runner.server_args):
            # Version-pinned (sglang 0.5.9): module-level prepare_mlp_sync_batch_raw
            # with attn_cp_size, no spec_algorithm / speculative_num_draft_tokens.
            # This is the single copy; the dflash path used to carry a stale
            # Scheduler.prepare_mlp_sync_batch_raw signature here.
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

    @torch.no_grad
    def _forward_extend(self, reqs):
        """Build the extend batch and run ONE capture forward; return logits_output.

        The single shared capture-forward primitive (both eagle3 and dflash route
        through here). Does NOT clear the pools — the caller clears after slicing,
        preserving the original ordering.
        """
        cache_params = CacheInitParams(
            disable=False,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            page_size=self.model_runner.server_args.page_size,
        )
        tree_cache = RadixCache(cache_params)

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        self._maybe_prepare_mlp_sync_batch(batch)
        # sglang 0.5.13+: prepare_for_extend stages input_ids on pinned CPU
        # (prefill_input_ids_cpu) and leaves batch.input_ids=None; the scheduler
        # normally materializes them to device via resolve_forward_inputs. We bypass
        # the scheduler, so perform that prefill H2D copy here.
        if getattr(batch, "prefill_input_ids_cpu", None) is not None:
            batch.input_ids = batch.prefill_input_ids_cpu.to(
                batch.device, non_blocking=True
            )
            batch.prefill_input_ids_cpu = None
        # sglang 0.5.13+: the ModelWorkerBatch step was removed. ForwardBatch.init_new
        # now consumes the ScheduleBatch directly and reads capture_hidden_mode from
        # it, so set it on the batch before init_new.
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        forward_batch = ForwardBatch.init_new(batch, self.model_runner)
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        output = self.model_runner.forward(forward_batch)
        return output.logits_output if hasattr(output, "logits_output") else output

    def _clear_pools(self):
        # TODO: can we not clear?
        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()

    def _set_eagle3_logits_flags(
        self, return_last_hidden_states: bool, return_logits: bool, shard_returns: bool
    ):
        # set the logits processor for the model runner
        for name, module in self.model_runner.model.named_modules():
            if isinstance(module, LogitsProcessorForEAGLE3):
                module.return_last_hidden_states = return_last_hidden_states
                module.return_logits = return_logits
                module.shard_returns = shard_returns

    # --- EAGLE3 extend (text) ----------------------------------------------

    @torch.no_grad
    def _extend_eagle3(
        self,
        reqs,
        capture_aux_hidden_states: bool = True,
        return_last_hidden_states: bool = False,
        return_logits: bool = False,
        shard_returns: bool = False,
    ):
        self._set_eagle3_logits_flags(
            return_last_hidden_states, return_logits, shard_returns
        )
        # sglang 0.5.13+: capture input lengths BEFORE the forward — prepare_for_extend
        # / forward release per-req fields (origin_input_ids becomes None afterwards).
        input_lens = [len(req.origin_input_ids) for req in reqs]
        eagle3_output = self._forward_extend(reqs)

        logits = eagle3_output.logits
        aux_hidden_states = eagle3_output.aux_hidden_states
        last_hidden_states = eagle3_output.last_hidden_states

        if shard_returns:
            tp_rank = dist.get_rank(get_tp_group())
            tp_size = dist.get_world_size(get_tp_group())
            batch_size = len(input_lens) // tp_size
            valid_indices = list(
                range(tp_rank * batch_size, (tp_rank + 1) * batch_size)
            )
            valid_input_lens = [input_lens[i] for i in valid_indices]

        if return_logits:
            if shard_returns:
                logits = _get_sharded_return(
                    logits, input_lens, valid_input_lens, valid_indices
                )
            else:
                logits = torch.split(logits, input_lens, dim=0)
        else:
            logits = [None] * len(reqs)

        if capture_aux_hidden_states:
            if shard_returns:
                aux_hidden_states = _get_sharded_return(
                    aux_hidden_states, input_lens, valid_input_lens, valid_indices
                )
            else:
                aux_hidden_states = torch.split(aux_hidden_states, input_lens, dim=0)
        else:
            aux_hidden_states = [None] * len(reqs)

        if return_last_hidden_states:
            if shard_returns:
                last_hidden_states = _get_sharded_return(
                    last_hidden_states, input_lens, valid_input_lens, valid_indices
                )
            else:
                last_hidden_states = torch.split(last_hidden_states, input_lens, dim=0)
        else:
            last_hidden_states = [None] * len(reqs)

        self._clear_pools()
        return logits, aux_hidden_states, last_hidden_states

    def extend(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        return_last_hidden_states: bool = False,
        return_logits: bool = True,
        shard_returns: bool = False,
    ):
        sampling_params = SamplingParams(temperature=0, max_new_tokens=1, top_k=1)
        reqs, data_cache = [], []

        if isinstance(input_ids, torch.Tensor):
            input_ids = torch.split(input_ids, 1, dim=0)
            attention_mask = torch.split(attention_mask, 1, dim=0)
            loss_mask = torch.split(loss_mask, 1, dim=0)

        for idx, (input_id_, attention_mask_, loss_mask_) in enumerate(
            zip(
                input_ids,
                attention_mask,
                loss_mask,
            )
        ):
            req = Req(
                rid=str(idx),
                origin_input_text="",
                origin_input_ids=input_id_.view(-1).tolist(),
                sampling_params=sampling_params,
            )
            # sglang 0.5.13+: Req.fill_ids was removed in favor of
            # full_untruncated_fill_ids (origin + output ids) plus an integer
            # fill_len, which the scheduler's PrefillAdder sets during admission.
            # We bypass the scheduler, so replicate that here with no prefix-cache
            # reuse (prefix_indices stays empty). prepare_for_extend asserts
            # fill_len - len(prefix_indices) == extend_input_len.
            req.full_untruncated_fill_ids = array("q", req.origin_input_ids)
            req.fill_len = len(req.full_untruncated_fill_ids)
            req.extend_input_len = req.fill_len - len(req.prefix_indices)
            req.logprob_start_len = len(req.origin_input_ids) - 1
            data_cache.append([input_id_, attention_mask_, loss_mask_])
            reqs.append(req)

        logits_list, aux_hidden_states_list, last_hidden_states_list = (
            self._extend_eagle3(
                reqs,
                capture_aux_hidden_states=True,
                return_last_hidden_states=return_last_hidden_states,
                return_logits=return_logits,
                shard_returns=shard_returns,
            )
        )

        return data_cache, logits_list, aux_hidden_states_list, last_hidden_states_list

    def extend_eagle3(self, *args, **kwargs):
        return self.extend(*args, **kwargs)

    # --- DFlash extend ------------------------------------------------------

    @torch.no_grad
    def extend_dflash(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ):
        """DFlash extend: capture the concatenated layer hidden states, no logits.

        Returns ``(data_cache, hidden_states_list)`` — the per-sample raw hidden
        states the DFlash engine stacks into a batch.
        """
        sampling_params = SamplingParams(temperature=0, max_new_tokens=1)
        reqs, data_cache = [], []

        if isinstance(input_ids, torch.Tensor):
            input_ids_list = torch.split(input_ids, 1, dim=0)
            attn_mask_list = torch.split(attention_mask, 1, dim=0)
            loss_mask_list = torch.split(loss_mask, 1, dim=0)

        for idx, (curr_ids, curr_attn, curr_loss) in enumerate(
            zip(input_ids_list, attn_mask_list, loss_mask_list)
        ):
            req = Req(
                rid=str(idx),
                origin_input_text="",
                origin_input_ids=curr_ids.view(-1).tolist(),
                sampling_params=sampling_params,
            )
            # sglang 0.5.13+: Req.fill_ids was removed in favor of
            # full_untruncated_fill_ids (origin + output ids) plus an integer
            # fill_len, which the scheduler's PrefillAdder sets during admission.
            # We bypass the scheduler, so replicate that here with no prefix-cache
            # reuse (prefix_indices stays empty). prepare_for_extend asserts
            # fill_len - len(prefix_indices) == extend_input_len.
            req.full_untruncated_fill_ids = array("q", req.origin_input_ids)
            req.fill_len = len(req.full_untruncated_fill_ids)
            req.extend_input_len = req.fill_len - len(req.prefix_indices)
            data_cache.append((curr_ids, curr_attn, curr_loss))
            reqs.append(req)

        # sglang 0.5.13+: capture input lengths BEFORE the forward (origin_input_ids
        # becomes None afterwards).
        input_lens = [len(req.origin_input_ids) for req in reqs]
        output = self._forward_extend(reqs)
        if (
            hasattr(output, "aux_hidden_states")
            and output.aux_hidden_states is not None
        ):
            hidden_states_list = torch.split(
                output.aux_hidden_states, input_lens, dim=0
            )
        elif hasattr(output, "hidden_states") and output.hidden_states is not None:
            hidden_states_list = torch.split(output.hidden_states, input_lens, dim=0)
        else:
            raise ValueError("SGLang output does not contain hidden states.")
        self._clear_pools()

        return data_cache, hidden_states_list


__all__ = ["SGLangCaptureBackend"]
