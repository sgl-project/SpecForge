"""
This file contains the wrapper for the SGL model.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn
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
    # When early_vocab_projection is applied inside the logits processor,
    # these fields carry the pre-projected metadata so that the caller
    # can skip the expensive full-vocab → draft-vocab projection later.
    logits_pre_projected: bool = False
    target_in_draft_mask: Optional[torch.Tensor] = None


# Default chunk size for chunked logits computation (number of tokens per chunk).
# Each chunk of 2048 tokens × 248K vocab × 2 bytes ≈ 0.96 GB peak memory.
# Set to 0 to disable chunking (original behavior).
LOGITS_CHUNK_SIZE = 2048

# Target-to-draft vocab mapping (bool tensor). When set, chunked logits will be
# projected to draft_vocab inside the logits processor, avoiding full-vocab cat.
# Set via set_early_projection_mapping().
# NOTE: kept for backward compatibility, but the preferred approach is to set
# t2d_mapping directly on LogitsProcessorForEAGLE3 instances.
_T2D_MAPPING: Optional[torch.Tensor] = None


def replaced_logits_processor_forward_for_eagle3(
    self,
    input_ids,
    hidden_states,
    lm_head,
    logits_metadata: Union[LogitsMetadata, ForwardBatch],
    aux_hidden_states: Optional[List[torch.Tensor]] = None,
    hidden_states_before_norm: Optional[torch.Tensor] = None,
    return_last_hidden_states: bool = False,
    return_logits: bool = False,
    t2d_mapping: Optional[torch.Tensor] = None,
    logits_chunk_size: Optional[int] = None,
) -> LogitsProcessorOutput:
    """
    This is a modified forward function for the SGLang's logits processor, adapted from https://github.com/sgl-project/sglang/blob/v0.5.4/python/sglang/srt/layers/logits_processor.py.
    The modification is to return the logits and aux hidden states instead of the last hidden states.

    Updated for sglang 0.5.9:
    - Added hidden_states_before_norm parameter for compatibility

    Memory optimization:
    - Chunked logits computation: when chunk_size > 0 and total tokens > chunk_size,
      computes logits in chunks to reduce peak all_gather memory from
      (total_tokens × vocab_size) to (chunk_size × vocab_size).
      For chunk_size=2048 and vocab_size=248K: peak ≈ 0.96 GB instead of
      e.g., 9.5 GB for a 20K sequence.

    Args:
        t2d_mapping: Bool tensor of shape (target_vocab_size,) for early projection.
                     If provided, takes priority over the module-level _T2D_MAPPING.
        logits_chunk_size: Chunk size override. If None, uses module-level LOGITS_CHUNK_SIZE.
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
        else:
            aux_pruned_states = None
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

    logits_pre_projected = False
    target_in_draft_mask = None

    if return_logits:
        # Compute logits — with optional chunking for long sequences to avoid OOM
        # during tensor_model_parallel_all_gather.
        # Prefer instance-level settings, fall back to module-level globals.
        chunk_size = logits_chunk_size if logits_chunk_size is not None else LOGITS_CHUNK_SIZE
        total_tokens = pruned_states.shape[0]
        t2d = t2d_mapping if t2d_mapping is not None else _T2D_MAPPING

        if chunk_size > 0 and total_tokens > chunk_size:
            if t2d is not None:
                # ---- Chunked + early projection path ----
                # Each chunk: compute full-vocab logits, immediately project to
                # draft_vocab, then discard full logits.
                # Peak memory: 1 chunk × full_vocab + accumulated projected logits.
                # E.g., chunk=2048, full_vocab=248K → 0.96 GiB peak per chunk;
                #        20K tokens × 32K draft_vocab → 1.2 GiB accumulated.
                #        Total peak ≈ 2.2 GiB vs 19 GiB without this optimization.
                draft_vocab_size = int(t2d.sum().item())
                projected_logits = torch.empty(
                    total_tokens, draft_vocab_size,
                    dtype=pruned_states.dtype, device=pruned_states.device,
                )
                all_in_draft_mask = torch.empty(
                    total_tokens, dtype=torch.bool, device=pruned_states.device,
                )

                for start in range(0, total_tokens, chunk_size):
                    end = min(start + chunk_size, total_tokens)
                    chunk_logits = self._get_logits(
                        pruned_states[start:end], lm_head, logits_metadata
                    )
                    # Compute target_in_draft_mask for this chunk
                    chunk_max_token = chunk_logits.argmax(-1)  # (chunk_len,)
                    all_in_draft_mask[start:end] = t2d[chunk_max_token]
                    # Project to draft vocab
                    projected_logits[start:end] = chunk_logits[..., t2d]
                    del chunk_logits, chunk_max_token

                logits = projected_logits
                target_in_draft_mask = all_in_draft_mask
                logits_pre_projected = True
            else:
                # ---- Chunked without projection ----
                # Pre-allocate and copy to avoid 2x peak from torch.cat
                first_chunk = self._get_logits(
                    pruned_states[:chunk_size], lm_head, logits_metadata
                )
                vocab_size = first_chunk.shape[-1]
                logits = torch.empty(
                    total_tokens, vocab_size,
                    dtype=first_chunk.dtype, device=first_chunk.device,
                )
                logits[:chunk_size] = first_chunk
                del first_chunk

                for start in range(chunk_size, total_tokens, chunk_size):
                    end = min(start + chunk_size, total_tokens)
                    chunk_logits = self._get_logits(
                        pruned_states[start:end], lm_head, logits_metadata
                    )
                    logits[start:end] = chunk_logits
                    del chunk_logits
        else:
            # Non-chunked path (short sequences)
            logits = self._get_logits(pruned_states, lm_head, logits_metadata)
            # Still apply early projection if t2d is set
            if t2d is not None:
                target_max_token = logits.argmax(-1)
                target_in_draft_mask = t2d[target_max_token]
                logits = logits[..., t2d]
                logits_pre_projected = True
                del target_max_token
    else:
        logits = None

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
        logits_pre_projected=logits_pre_projected,
        target_in_draft_mask=target_in_draft_mask,
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
        # Instance-level t2d mapping and chunk size (set via set_vocab_mapping /
        # set_logits_chunk_size on SGLangEagle3TargetModel).  These take priority
        # over the module-level globals, eliminating any risk of the global not
        # being visible due to import-path or multi-process issues.
        self.t2d_mapping: Optional[torch.Tensor] = None
        self.logits_chunk_size: Optional[int] = None

    def forward(
        self,
        input_ids,
        hidden_states,
        lm_head,
        logits_metadata,
        aux_hidden_states: Optional[torch.Tensor] = None,
        hidden_states_before_norm: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorOutput:
        logits_metadata.forward_mode = ForwardMode.DECODE
        ret = replaced_logits_processor_forward_for_eagle3(
            self.logits_processor,
            input_ids,
            hidden_states,
            lm_head,
            logits_metadata,
            aux_hidden_states,
            hidden_states_before_norm,
            self.return_last_hidden_states,
            self.return_logits,
            t2d_mapping=self.t2d_mapping,
            logits_chunk_size=self.logits_chunk_size,
        )
        return ret


def wrap_eagle3_logits_processors_in_module(
    module: nn.Module, return_full_logits: bool = False
):
    """
    This function will wrap the SGLang's original logits processor with the modified one for EAGLE3.
    """
    for name, submodule in module.named_modules():
        if isinstance(submodule, LogitsProcessor):
            wrapped = LogitsProcessorForEAGLE3(submodule, return_full_logits)
            # Handle nested module paths: named_modules() may return dotted names
            # like "model.logits_processor". setattr(module, dotted_name, ...) would
            # create a literal attribute with a dot in its name rather than setting
            # the nested attribute.  Walk the path to set correctly.
            parts = name.split(".")
            parent = module
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], wrapped)
            print(f"wrapped {name} with LogitsProcessorForEAGLE3")
