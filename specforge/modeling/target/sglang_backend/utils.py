"""
This file contains the wrapper for the SGL model.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
from sglang.srt.distributed.communication_op import tensor_model_parallel_all_gather
from sglang.srt.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.server_args import get_global_server_args


@dataclass
class ReplacedLogitsProcessorEagle3Output:
    """
    A dataclass to store the logits and aux hidden states needed for EAGLE3.
    """

    logits: torch.Tensor
    aux_hidden_states: torch.Tensor
    last_hidden_states: Optional[torch.Tensor] = None
    logits_are_tp_sharded: bool = False


def replaced_logits_processor_forward_for_eagle3(
    self,
    input_ids,
    hidden_states,
    lm_head,
    logits_metadata: Union[LogitsMetadata, ForwardBatch],
    aux_hidden_states: Optional[torch.Tensor] = None,
    return_last_hidden_states: bool = False,
    return_logits: bool = False,
) -> LogitsProcessorOutput:
    """
    This is a modified forward function for the SGLang's logits processor, adapted from https://github.com/sgl-project/sglang/blob/v0.5.4/python/sglang/srt/layers/logits_processor.py.
    The modification is to return the logits and aux hidden states instead of the last hidden states.
    """

    if isinstance(logits_metadata, ForwardBatch):
        logits_metadata = LogitsMetadata.from_forward_batch(logits_metadata)

    # Check if multi-item scoring is enabled via server args (only for prefill-only requests)
    multi_item_delimiter = get_global_server_args().multi_item_scoring_delimiter
    if multi_item_delimiter is not None and logits_metadata.is_prefill_only:
        return self.compute_logprobs_for_multi_item_scoring(
            input_ids, hidden_states, lm_head, logits_metadata, multi_item_delimiter
        )

    # Get the last hidden states and last logits for the next token prediction
    if (
        logits_metadata.forward_mode.is_decode_or_idle()
        or logits_metadata.forward_mode.is_target_verify()
        or logits_metadata.forward_mode.is_draft_extend_v2()
    ):
        pruned_states = hidden_states
        if aux_hidden_states is not None:
            aux_pruned_states = [hidden for hidden in aux_hidden_states]
        sample_indices = None
        input_logprob_indices = None
    else:
        raise RuntimeError(
            f"The modified logits processor is not supported for this forward mode: {logits_metadata.forward_mode}"
        )

    if return_last_hidden_states:
        last_hidden_states = pruned_states
    else:
        last_hidden_states = None

    if return_logits:
        # Compute logits in chunks to avoid OOM on large-vocabulary models.
        # Logits are kept TP-sharded (vocab_size / tp_size per rank) to avoid
        # materializing the full [total_tokens, vocab_size] tensor which can be
        # ~40 GB for models with large vocabularies (e.g., 163840 for Kimi-K2.5).
        # The all-gather is deferred to per-sequence processing in generate_eagle3_data.
        chunk_size = 16384
        num_tokens = pruned_states.shape[0]
        logits_chunks = []
        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)
            chunk_logits = self._compute_lm_head(
                pruned_states[start:end], lm_head, None
            )
            if self.logit_scale is not None:
                chunk_logits.mul_(self.logit_scale)
            chunk_logits = chunk_logits.bfloat16()
            logits_chunks.append(chunk_logits)
        logits = torch.cat(logits_chunks, dim=0) if len(logits_chunks) > 1 else logits_chunks[0]
        logits_tp_sharded = self.do_tensor_parallel_all_gather
    else:
        logits = None
        logits_tp_sharded = False

    # get the aux hidden states
    hidden_states_to_store: Optional[torch.Tensor] = None
    if logits_metadata.capture_hidden_mode.need_capture():
        if logits_metadata.capture_hidden_mode.is_full():
            if aux_hidden_states is not None:
                aux_hidden_states = torch.cat(aux_hidden_states, dim=-1)
                hidden_states_to_store = aux_hidden_states
            else:
                hidden_states_to_store = hidden_states
        elif logits_metadata.capture_hidden_mode.is_last():
            # Get the last token hidden states. If sample_indices is None,
            # pruned states only contain the last tokens already.
            if aux_hidden_states is not None:
                aux_pruned_states = torch.cat(aux_pruned_states, dim=-1)
                hidden_states_to_store = (
                    aux_pruned_states[sample_indices]
                    if sample_indices is not None
                    else aux_pruned_states
                )
            else:
                hidden_states_to_store = (
                    pruned_states[sample_indices]
                    if sample_indices is not None
                    else pruned_states
                )
        else:
            assert False, "Should never reach"

    assert (
        not logits_metadata.extend_return_logprob
    ), "extend_return_logprob is not supported"
    # Decode mode or extend mode without return_logprob.
    return ReplacedLogitsProcessorEagle3Output(
        logits=logits,
        aux_hidden_states=hidden_states_to_store,
        last_hidden_states=last_hidden_states,
        logits_are_tp_sharded=logits_tp_sharded,
    )


class LogitsProcessorForEAGLE3(torch.nn.Module):
    def __init__(
        self,
        logits_processor: LogitsProcessor,
        return_last_hidden_states: bool = False,
        return_logits: bool = False,
    ):
        super().__init__()
        self.logits_processor = logits_processor
        self.return_last_hidden_states = return_last_hidden_states
        self.return_logits = return_logits

    def forward(
        self,
        input_ids,
        hidden_states,
        lm_head,
        logits_metadata,
        aux_hidden_states: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorOutput:
        logits_metadata.forward_mode = ForwardMode.DECODE
        ret = replaced_logits_processor_forward_for_eagle3(
            self.logits_processor,
            input_ids,
            hidden_states,
            lm_head,
            logits_metadata,
            aux_hidden_states,
            self.return_last_hidden_states,
            self.return_logits,
        )
        return ret


def wrap_eagle3_logits_processors_in_module(
    module: nn.Module, return_full_logits: bool = False
):
    """
    This function will wrap the SGLang's original logits processor with the modified one for EAGLE3.
    """
    to_wrap = [
        (name, submodule)
        for name, submodule in module.named_modules()
        if isinstance(submodule, LogitsProcessor)
    ]
    for name, submodule in to_wrap:
        wrapped = LogitsProcessorForEAGLE3(submodule, return_full_logits)
        # Navigate the module hierarchy for nested names (e.g. "language_model.logits_processor")
        parts = name.split(".")
        parent = module
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], wrapped)
        print(f"wrapped {name} with LogitsProcessorForEAGLE3")
