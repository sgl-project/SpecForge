"""
This file contains the wrapper for the SGL model.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.distributed as dist
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
    local_sample_indices: Optional[list] = None
    local_input_lens: Optional[list] = None


def all_to_all_batch_sharded_logits(
    logits: torch.Tensor,
    group: dist.ProcessGroup,
    input_lens: list,
) -> tuple:
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    num_samples = len(input_lens)
    local_vocab_size = logits.shape[1]
    max_seq_len = max(input_lens)

    if world_size == 1:
        return logits, input_lens, list(range(num_samples))

    padded_batch_size = ((num_samples + world_size - 1) // world_size) * world_size
    num_pad_samples = padded_batch_size - num_samples
    padded_logits = torch.zeros(
        (padded_batch_size, max_seq_len, local_vocab_size),
        dtype=logits.dtype,
        device=logits.device,
    )

    token_offset = 0
    for sample_idx, seq_len in enumerate(input_lens):
        padded_logits[sample_idx, :seq_len] = logits[
            token_offset : token_offset + seq_len
        ]
        token_offset += seq_len

    padded_input_lens = input_lens + [0] * num_pad_samples
    samples_per_rank = padded_batch_size // world_size
    input_list = list(padded_logits.chunk(world_size, dim=0))
    output_buf = torch.empty(
        (world_size, samples_per_rank, max_seq_len, local_vocab_size),
        dtype=logits.dtype,
        device=logits.device,
    )
    output_list = list(output_buf.unbind(0))
    dist.all_to_all(output_list, input_list, group=group)
    local_logits_padded = torch.cat(output_list, dim=-1)

    start_sample = rank * samples_per_rank
    end_sample = min(start_sample + samples_per_rank, num_samples)
    local_sample_indices = list(range(start_sample, end_sample))
    local_input_lens = padded_input_lens[start_sample:end_sample]

    local_tokens = []
    for sample_idx, seq_len in enumerate(local_input_lens):
        if seq_len > 0:
            local_tokens.append(local_logits_padded[sample_idx, :seq_len])

    if not local_tokens:
        return (
            torch.empty(
                (0, local_logits_padded.shape[-1]),
                dtype=logits.dtype,
                device=logits.device,
            ),
            [],
            [],
        )

    return (
        torch.cat(local_tokens, dim=0),
        [seq_len for seq_len in local_input_lens if seq_len > 0],
        local_sample_indices,
    )


def all_to_all_sequence_sharded_logits(
    logits: torch.Tensor,
    group: dist.ProcessGroup,
    input_lens: list,
    ttt_length: int,
    sequence_rank: int,
    sequence_size: int,
) -> tuple:
    world_size = dist.get_world_size(group)
    num_samples = len(input_lens)
    local_vocab_size = logits.shape[1]

    if world_size == 1:
        return logits, input_lens, list(range(num_samples))

    if world_size % sequence_size != 0:
        raise ValueError(
            f"TP size ({world_size}) must be divisible by SP size ({sequence_size}) "
            "when sharding target logits for sequence parallel training."
        )
    if sequence_rank < 0 or sequence_rank >= sequence_size:
        raise ValueError(
            f"sequence_rank must be in [0, {sequence_size}), got {sequence_rank}."
        )

    max_seq_len = max(input_lens)
    chunk_size = (max_seq_len + sequence_size - 1) // sequence_size
    local_len = chunk_size + ttt_length
    padded_logits = torch.zeros(
        (num_samples, max_seq_len, local_vocab_size),
        dtype=logits.dtype,
        device=logits.device,
    )

    token_offset = 0
    for sample_idx, seq_len in enumerate(input_lens):
        padded_logits[sample_idx, :seq_len] = logits[
            token_offset : token_offset + seq_len
        ]
        token_offset += seq_len

    sequence_rank_tensor = torch.tensor(
        [sequence_rank], dtype=torch.int64, device=logits.device
    )
    gathered_sequence_ranks = [
        torch.empty_like(sequence_rank_tensor) for _ in range(world_size)
    ]
    dist.all_gather(gathered_sequence_ranks, sequence_rank_tensor, group=group)

    send_buf = torch.zeros(
        (world_size, num_samples, local_len, local_vocab_size),
        dtype=logits.dtype,
        device=logits.device,
    )
    for dst_rank in range(world_size):
        dst_sequence_rank = int(gathered_sequence_ranks[dst_rank].item())
        start = dst_sequence_rank * chunk_size
        for sample_idx, seq_len in enumerate(input_lens):
            end = min(start + local_len, seq_len)
            valid_len = max(0, end - start)
            if valid_len > 0:
                send_buf[dst_rank, sample_idx, :valid_len] = padded_logits[
                    sample_idx, start:end
                ]

    input_list = list(send_buf.unbind(0))
    output_buf = torch.empty(
        (world_size, num_samples, local_len, local_vocab_size),
        dtype=logits.dtype,
        device=logits.device,
    )
    output_list = list(output_buf.unbind(0))
    dist.all_to_all(output_list, input_list, group=group)
    local_logits_padded = torch.cat(output_list, dim=-1)
    return (
        local_logits_padded.reshape(
            num_samples * local_len, local_logits_padded.shape[-1]
        ),
        [local_len] * num_samples,
        list(range(num_samples)),
    )


def slice_hidden_states_by_samples(
    hidden_states: torch.Tensor,
    input_lens: list,
    local_sample_indices: list,
) -> torch.Tensor:
    if not local_sample_indices:
        return torch.empty(
            (0, hidden_states.shape[1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    sample_offsets = [0]
    for seq_len in input_lens:
        sample_offsets.append(sample_offsets[-1] + seq_len)

    local_tokens = []
    for sample_idx in local_sample_indices:
        start = sample_offsets[sample_idx]
        end = sample_offsets[sample_idx + 1]
        local_tokens.append(hidden_states[start:end])
    return torch.cat(local_tokens, dim=0)


def slice_hidden_states_by_sequence_chunks(
    hidden_states: torch.Tensor,
    input_lens: list,
    sequence_rank: int,
    sequence_size: int,
    ttt_length: int,
) -> torch.Tensor:
    if sequence_size == 1:
        return hidden_states

    max_seq_len = max(input_lens)
    chunk_size = (max_seq_len + sequence_size - 1) // sequence_size
    local_len = chunk_size + ttt_length
    sample_offsets = [0]
    for seq_len in input_lens:
        sample_offsets.append(sample_offsets[-1] + seq_len)

    local_tokens = torch.zeros(
        (len(input_lens), local_len, hidden_states.shape[1]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    start = sequence_rank * chunk_size
    for sample_idx, seq_len in enumerate(input_lens):
        end = min(start + local_len, seq_len)
        valid_len = max(0, end - start)
        if valid_len > 0:
            sample_start = sample_offsets[sample_idx]
            local_tokens[sample_idx, :valid_len] = hidden_states[
                sample_start + start : sample_start + end
            ]

    return local_tokens.reshape(len(input_lens) * local_len, hidden_states.shape[1])


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
    shard_target_logits: bool = False,
    sequence_parallel: bool = False,
    sequence_rank: int = 0,
    sequence_size: int = 1,
    sp_ttt_length: int = 0,
    tp_group: Optional[dist.ProcessGroup] = None,
) -> LogitsProcessorOutput:
    """
    This is a modified forward function for the SGLang's logits processor, adapted from https://github.com/sgl-project/sglang/blob/v0.5.4/python/sglang/srt/layers/logits_processor.py.
    The modification is to return the logits and aux hidden states instead of the last hidden states.

    Updated for sglang 0.5.9:
    - Added hidden_states_before_norm parameter for compatibility
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

    if hasattr(logits_metadata, "seq_lens"):
        all_input_lens = list(logits_metadata.seq_lens)
    elif hasattr(logits_metadata, "extend_seq_lens"):
        all_input_lens = list(logits_metadata.extend_seq_lens)
    else:
        all_input_lens = [hidden_states.shape[0]]

    local_sample_indices = None
    local_input_lens = None

    if return_logits:
        original_do_all_gather = self.do_tensor_parallel_all_gather
        if (
            shard_target_logits
            and tp_group is not None
            and dist.get_world_size(tp_group) > 1
        ):
            self.do_tensor_parallel_all_gather = False

        logits = self._get_logits(pruned_states, lm_head, logits_metadata)
        if (
            shard_target_logits
            and tp_group is not None
            and dist.get_world_size(tp_group) > 1
        ):
            if sequence_parallel:
                # Convert sequence-major local-vocab logits into local-sequence full-vocab logits.
                logits, local_input_lens, local_sample_indices = (
                    all_to_all_sequence_sharded_logits(
                        logits,
                        tp_group,
                        all_input_lens,
                        sp_ttt_length,
                        sequence_rank,
                        sequence_size,
                    )
                )
            else:
                # Convert batch-major local-vocab logits into local-batch full-vocab logits.
                logits, local_input_lens, local_sample_indices = (
                    all_to_all_batch_sharded_logits(
                        logits, tp_group, all_input_lens
                    )
                )
        else:
            local_input_lens = all_input_lens
            local_sample_indices = list(range(len(all_input_lens)))
        self.do_tensor_parallel_all_gather = original_do_all_gather
    else:
        logits = None
        local_input_lens = all_input_lens
        local_sample_indices = list(range(len(all_input_lens)))

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

        if (
            shard_target_logits
            and hidden_states_to_store is not None
            and tp_group is not None
            and dist.get_world_size(tp_group) > 1
        ):
            if sequence_parallel:
                hidden_states_to_store = slice_hidden_states_by_sequence_chunks(
                    hidden_states_to_store,
                    all_input_lens,
                    sequence_rank,
                    sequence_size,
                    sp_ttt_length,
                )
            else:
                hidden_states_to_store = slice_hidden_states_by_samples(
                    hidden_states_to_store, all_input_lens, local_sample_indices
                )

    if (
        shard_target_logits
        and last_hidden_states is not None
        and tp_group is not None
        and dist.get_world_size(tp_group) > 1
    ):
        if sequence_parallel:
            last_hidden_states = slice_hidden_states_by_sequence_chunks(
                last_hidden_states,
                all_input_lens,
                sequence_rank,
                sequence_size,
                sp_ttt_length,
            )
        else:
            last_hidden_states = slice_hidden_states_by_samples(
                last_hidden_states, all_input_lens, local_sample_indices
            )

    assert (
        not logits_metadata.extend_return_logprob
    ), "extend_return_logprob is not supported"
    # Decode mode or extend mode without return_logprob.
    return ReplacedLogitsProcessorEagle3Output(
        logits=logits,
        aux_hidden_states=hidden_states_to_store,
        last_hidden_states=last_hidden_states,
        local_sample_indices=local_sample_indices,
        local_input_lens=local_input_lens,
    )


class LogitsProcessorForEAGLE3(torch.nn.Module):
    def __init__(
        self,
        logits_processor: LogitsProcessor,
        return_last_hidden_states: bool = False,
        return_logits: bool = False,
        shard_target_logits: bool = False,
        sequence_parallel: bool = False,
        sequence_rank: int = 0,
        sequence_size: int = 1,
        sp_ttt_length: int = 0,
        tp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.logits_processor = logits_processor
        self.return_last_hidden_states = return_last_hidden_states
        self.return_logits = return_logits
        self.shard_target_logits = shard_target_logits
        self.sequence_parallel = sequence_parallel
        self.sequence_rank = sequence_rank
        self.sequence_size = sequence_size
        self.sp_ttt_length = sp_ttt_length
        self.tp_group = tp_group

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
            self.shard_target_logits,
            self.sequence_parallel,
            self.sequence_rank,
            self.sequence_size,
            self.sp_ttt_length,
            self.tp_group,
        )
        return ret


def wrap_eagle3_logits_processors_in_module(
    module: nn.Module,
    return_full_logits: bool = False,
    shard_target_logits: bool = False,
    tp_group: Optional[dist.ProcessGroup] = None,
):
    """
    This function will wrap the SGLang's original logits processor with the modified one for EAGLE3.
    """
    for name, submodule in module.named_modules():
        if isinstance(submodule, LogitsProcessor):
            wrapped = LogitsProcessorForEAGLE3(
                submodule,
                return_last_hidden_states=False,
                return_logits=return_full_logits,
                shard_target_logits=shard_target_logits,
                tp_group=tp_group,
            )
            setattr(module, name, wrapped)
            print(
                f"wrapped {name} with LogitsProcessorForEAGLE3 "
                f"(shard_target_logits={shard_target_logits})"
            )
