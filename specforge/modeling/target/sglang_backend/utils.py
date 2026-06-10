"""
This file contains the wrapper for the SGL model.
"""

from dataclasses import dataclass
from typing import Optional, Union, cast

import torch
import torch.nn as nn
from sglang.srt.distributed import (
    GroupCoordinator,
    get_tp_group,
    tensor_model_parallel_all_gather,
)
from sglang.srt.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
    fused_softcap,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils.common import is_npu


@dataclass
class ReplacedLogitsProcessorEagle3Output:
    """
    A dataclass to store the logits and aux hidden states needed for EAGLE3.
    """

    logits: torch.Tensor
    aux_hidden_states: torch.Tensor
    last_hidden_states: Optional[torch.Tensor] = None


def replaced_logits_processor_forward_for_eagle3(
    self,
    input_ids,
    hidden_states,
    lm_head,
    logits_metadata: Union[LogitsMetadata, ForwardBatch],
    aux_hidden_states: Optional[list[torch.Tensor]] = None,
    hidden_states_before_norm: Optional[torch.Tensor] = None,
    return_last_hidden_states: bool = False,
    return_logits: bool = False,
    shard_returns: bool = False,
) -> LogitsProcessorOutput:
    """
    This is a modified forward function for the SGLang's logits processor, adapted from https://github.com/sgl-project/sglang/blob/v0.5.9/python/sglang/srt/layers/logits_processor.py.
    The modification is to return the logits and aux hidden states instead of the last hidden states.

    Updated for sglang 0.5.9:
    - Added hidden_states_before_norm parameter for compatibility
    """
    if isinstance(logits_metadata, ForwardBatch):
        logits_metadata = LogitsMetadata.from_forward_batch(logits_metadata)

    # Multi-item scoring only for prefill-only requests.
    if self.multi_item_delimiter is not None and logits_metadata.is_prefill_only:
        return self.compute_logprobs_for_multi_item_scoring(
            input_ids,
            hidden_states,
            lm_head,
            logits_metadata,
            self.multi_item_delimiter,
        )

    # Diffusion LLM only.
    if logits_metadata.forward_mode.is_dllm_extend():
        raise RuntimeError(
            f"The modified logits processor is not supported for this forward mode: {logits_metadata.forward_mode}"
        )

    # Get the last hidden states and last logits for the next token prediction
    if not (
        logits_metadata.forward_mode.is_decode_or_idle()
        or logits_metadata.forward_mode.is_target_verify()
        or logits_metadata.forward_mode.is_draft_extend_v2()
    ):
        raise RuntimeError(
            f"The modified logits processor is not supported for this forward mode: {logits_metadata.forward_mode}"
        )
    (
        pruned_states,
        pruned_states_before_norm,
        aux_pruned_states,
        sample_indices,
        _,
        _,
    ) = self._get_pruned_states(
        hidden_states,
        hidden_states_before_norm,
        aux_hidden_states,
        logits_metadata,
    )

    tp_group: Optional[GroupCoordinator] = None
    chunk_sizes: list[int] = []
    if shard_returns:
        tp_group = get_tp_group()
        seq_lens = logits_metadata.extend_seq_lens.tolist()
        if len(seq_lens) != tp_group.world_size:
            assert len(seq_lens) % tp_group.world_size == 0
            size = len(seq_lens) // tp_group.world_size
            chunk_sizes = []
            for i in range(0, len(seq_lens), size):
                chunk_sizes.append(sum(seq_lens[i : i + size]))
        else:
            chunk_sizes = seq_lens

    if return_last_hidden_states:
        last_hidden_states = pruned_states
        if shard_returns:
            last_hidden_states = torch.split(last_hidden_states, chunk_sizes, dim=0)[
                cast(GroupCoordinator, tp_group).rank_in_group
            ]
    else:
        last_hidden_states = None

    if return_logits:
        # Compute logits for both input and sampled tokens.
        logits = replaced_logits_processor_get_logits(
            self,
            pruned_states,
            lm_head,
            logits_metadata,
            chunk_sizes=chunk_sizes if shard_returns else None,
        )
    else:
        logits = None

    if shard_returns:
        if aux_hidden_states is not None:
            aux_hidden_states = [
                torch.split(hidden, chunk_sizes, dim=0)[
                    cast(GroupCoordinator, tp_group).rank_in_group
                ]
                for hidden in aux_hidden_states
            ]
        hidden_states = torch.split(hidden_states, chunk_sizes, dim=0)[
            cast(GroupCoordinator, tp_group).rank_in_group
        ]

    hidden_states_to_store = self._get_hidden_states_to_store(
        hidden_states,
        hidden_states_before_norm,
        aux_hidden_states,
        pruned_states,
        pruned_states_before_norm,
        aux_pruned_states,
        sample_indices,
        logits_metadata,
    )
    del hidden_states

    assert (
        not logits_metadata.extend_return_logprob
    ), "extend_return_logprob is not supported"
    # Decode mode or extend mode without return_logprob.
    return ReplacedLogitsProcessorEagle3Output(
        logits=logits,
        aux_hidden_states=hidden_states_to_store,
        last_hidden_states=last_hidden_states,
    )


def replaced_logits_processor_get_logits(
    self,
    hidden_states: torch.Tensor,
    lm_head: VocabParallelEmbedding,
    logits_metadata: LogitsMetadata,
    embedding_bias: Optional[torch.Tensor] = None,
    chunk_sizes: Optional[list[int]] = None,
    group: Optional[GroupCoordinator] = None,
) -> torch.Tensor:
    """
    This is a modified forward function for the SGLang's logits processor, adapted from https://github.com/sgl-project/sglang/blob/v0.5.9/python/sglang/srt/layers/logits_processor.py.
    The modification is to use all_to_all instead of gather to reduce memory footprint.
    """
    hidden_states, local_hidden_states = self._gather_dp_attn_hidden_states(
        hidden_states, logits_metadata
    )

    logits = self._compute_lm_head(hidden_states, lm_head, embedding_bias)

    if self.logit_scale is not None:
        logits.mul_(self.logit_scale)

    if self.do_tensor_parallel_all_gather:
        if chunk_sizes is not None:
            if self.use_attn_tp_group:
                raise NotImplementedError(
                    "'shard_returns' does not support attention tensor parallel"
                )
            else:
                logits = tensor_all_to_all(logits, chunk_sizes, group=group)
        else:
            if self.use_attn_tp_group:
                logits = self._gather_attn_tp_logits(logits)
            else:
                logits = tensor_model_parallel_all_gather(logits)

    logits = self._scatter_dp_attn_logits(logits, local_hidden_states, logits_metadata)

    logits = self._copy_logits_to_buffer(logits, logits_metadata)

    if self.final_logit_softcapping:
        if not is_npu():
            fused_softcap(logits, self.final_logit_softcapping)
        else:
            logits = self.final_logit_softcapping * torch.tanh(
                logits / self.final_logit_softcapping
            )

    return logits


class LogitsProcessorForEAGLE3(torch.nn.Module):
    def __init__(
        self,
        logits_processor: LogitsProcessor,
        return_last_hidden_states: bool = False,
        return_logits: bool = False,
        shard_returns: bool = False,
    ):
        super().__init__()
        self.logits_processor = logits_processor
        self.return_last_hidden_states = return_last_hidden_states
        self.return_logits = return_logits
        self.shard_returns = shard_returns

    @property
    def shard_returns(self):
        return self._shard_returns

    @shard_returns.setter
    def shard_returns(self, v):
        self._shard_returns = v
        # NOTE(@cih9088): `shard_returns` does not cover all control path.
        if self._shard_returns and self.logits_processor.use_attn_tp_group:
            raise NotImplementedError(
                "'shard_returns' does not support attention tensor parallel"
            )
        if (
            not self.logits_processor.do_tensor_parallel_all_gather
            and self._shard_returns
        ):
            raise ValueError(
                "'shard_returns' does nothing if do_tensor_parallel_all_gather is False"
            )

    def forward(
        self,
        input_ids,
        hidden_states,
        lm_head,
        logits_metadata,
        aux_hidden_states: Optional[list[torch.Tensor]] = None,
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
            self.shard_returns,
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
            setattr(module, name, wrapped)
            print(f"wrapped {name} with LogitsProcessorForEAGLE3")


def tensor_all_to_all(
    input_: torch.Tensor,
    chunk_sizes: Optional[list[int]] = None,
    scatter_dim: int = 0,
    gather_dim: int = -1,
    group: Optional[GroupCoordinator] = None,
):
    group = group if group is not None else get_tp_group()
    if chunk_sizes is None:
        assert input_.shape[scatter_dim] % group.world_size == 0
        chunk_size = input_.shape[scatter_dim] // group.world_size
        chunk_sizes = [chunk_size for _ in range(group.world_size)]

    assert group.world_size == len(chunk_sizes), "chunk size must equal to world size"

    scatter_list = list(input_.split(chunk_sizes, dim=scatter_dim))
    gather_list = [
        torch.zeros_like(scatter_list[group.rank_in_group]) for _ in scatter_list
    ]
    torch.distributed.all_to_all(gather_list, scatter_list, group=group.device_group)
    return torch.cat(gather_list, dim=gather_dim)
