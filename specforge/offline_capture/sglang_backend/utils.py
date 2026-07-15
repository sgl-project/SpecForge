"""Small SGLang logits-processor patch for offline EAGLE3 state capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


@dataclass
class OfflineEagle3LogitsOutput:
    """Only the tensors consumed by offline hidden-state preparation."""

    logits: None
    aux_hidden_states: torch.Tensor
    last_hidden_states: torch.Tensor


def replaced_logits_processor_forward_for_offline_eagle3(
    logits_processor: LogitsProcessor,
    input_ids,
    hidden_states,
    lm_head,
    logits_metadata,
    aux_hidden_states: Optional[list[torch.Tensor]] = None,
    hidden_states_before_norm: Optional[torch.Tensor] = None,
):
    """Return full auxiliary/final states while skipping the LM-head projection."""

    multi_item_delimiter_indices = None
    if isinstance(logits_metadata, ForwardBatch):
        multi_item_delimiter_indices = logits_metadata.multi_item_delimiter_indices
        logits_metadata = LogitsMetadata.from_forward_batch(logits_metadata)

    if multi_item_delimiter_indices is not None and logits_metadata.is_prefill_only:
        return logits_processor.compute_logprobs_for_multi_item_scoring(
            input_ids,
            hidden_states,
            lm_head,
            logits_metadata,
            multi_item_delimiter_indices,
        )

    if logits_metadata.forward_mode.is_dllm_extend():
        raise RuntimeError(
            "Offline EAGLE3 capture does not support diffusion-LLM forward mode"
        )
    if not (
        logits_metadata.forward_mode.is_decode_or_idle()
        or logits_metadata.forward_mode.is_target_verify()
        or logits_metadata.forward_mode.is_draft_extend_v2()
    ):
        raise RuntimeError(
            "Offline EAGLE3 capture received an unsupported SGLang forward mode: "
            f"{logits_metadata.forward_mode}"
        )

    (
        pruned_states,
        pruned_states_before_norm,
        aux_pruned_states,
        sample_indices,
        _,
        _,
    ) = logits_processor._get_pruned_states(
        hidden_states,
        hidden_states_before_norm,
        aux_hidden_states,
        logits_metadata,
    )
    states_to_store = logits_processor._get_hidden_states_to_store(
        hidden_states,
        hidden_states_before_norm,
        aux_hidden_states,
        pruned_states,
        pruned_states_before_norm,
        aux_pruned_states,
        sample_indices,
        logits_metadata,
    )
    if logits_metadata.extend_return_logprob:
        raise RuntimeError("Offline EAGLE3 capture does not support log probabilities")
    return OfflineEagle3LogitsOutput(
        logits=None,
        aux_hidden_states=states_to_store,
        last_hidden_states=pruned_states,
    )


class OfflineEagle3LogitsProcessor(nn.Module):
    """SGLang logits processor that returns hidden states and computes no logits."""

    def __init__(self, logits_processor: LogitsProcessor) -> None:
        super().__init__()
        self.logits_processor = logits_processor

    def forward(
        self,
        input_ids,
        hidden_states,
        lm_head,
        logits_metadata,
        aux_hidden_states: Optional[list[torch.Tensor]] = None,
        hidden_states_before_norm: Optional[torch.Tensor] = None,
    ):
        # EAGLE3 needs every token, so use the decode pruning behavior for this
        # one offline extend pass rather than SGLang's normal prefill pruning.
        logits_metadata.forward_mode = ForwardMode.DECODE
        return replaced_logits_processor_forward_for_offline_eagle3(
            self.logits_processor,
            input_ids,
            hidden_states,
            lm_head,
            logits_metadata,
            aux_hidden_states,
            hidden_states_before_norm,
        )


def wrap_offline_eagle3_logits_processors(module: nn.Module) -> None:
    """Replace each SGLang logits processor with the offline capture variant."""

    replacements = [
        (name, submodule)
        for name, submodule in module.named_modules()
        if isinstance(submodule, LogitsProcessor)
    ]
    if not replacements:
        raise RuntimeError("No SGLang logits processor was found on the target model")
    for name, submodule in replacements:
        if not name:
            raise RuntimeError("The target model itself cannot be a logits processor")
        parent_name, _, child_name = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, child_name, OfflineEagle3LogitsProcessor(submodule))


__all__ = [
    "OfflineEagle3LogitsOutput",
    "OfflineEagle3LogitsProcessor",
    "replaced_logits_processor_forward_for_offline_eagle3",
    "wrap_offline_eagle3_logits_processors",
]
