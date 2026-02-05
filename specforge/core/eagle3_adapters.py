from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn

from specforge.distributed import get_draft_sp_group, get_sp_ulysses_group


@dataclass
class StepState:
    input_ids: torch.Tensor
    hidden_states: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_p: torch.Tensor
    position_mask: torch.Tensor
    loss_mask: torch.Tensor


class BackendAdapter:
    def __init__(self, model: "OnlineEagle3Model"):
        self.m = model

    def step_view(
        self,
        *,
        idx: int,
        ttt_length: int,
        global_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        target_p_padded: torch.Tensor,
        position_mask: torch.Tensor,
        seq_length: int,
    ) -> StepState:
        raise NotImplementedError

    def reduce_metrics(
        self, *, local_correct: torch.Tensor, local_denom: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return local_correct, local_denom

    def reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss


class SdpaLikeAdapter(BackendAdapter):
    def step_view(
        self,
        *,
        idx: int,
        ttt_length: int,
        global_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        target_p_padded: torch.Tensor,
        position_mask: torch.Tensor,
        seq_length: int,
    ) -> StepState:
        target_p = target_p_padded[:, idx : idx + seq_length, :].contiguous()
        return StepState(
            input_ids=global_input_ids,
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            target_p=target_p,
            position_mask=position_mask,
            loss_mask=loss_mask,
        )


class UspAdapter(BackendAdapter):
    def __init__(self, model: "OnlineEagle3Model"):
        super().__init__(model)
        self.sp_group = get_draft_sp_group()
        self.sp_world_size = dist.get_world_size(self.sp_group)
        self.ulysses_pg = get_sp_ulysses_group()
        self.sp_ulysses_degree = dist.get_world_size(self.ulysses_pg)

    def step_view(
        self,
        *,
        idx: int,
        ttt_length: int,
        global_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        target_p_padded: torch.Tensor,
        position_mask: torch.Tensor,
        seq_length: int,
    ) -> StepState:
        usp_chunk_size = seq_length - ttt_length
        if usp_chunk_size <= 0:
            raise ValueError(
                f"USP local seq_length ({seq_length}) must be larger than "
                f"ttt_length ({ttt_length})"
            )
        target_p = target_p_padded[:, idx : idx + usp_chunk_size, :]
        return StepState(
            input_ids=global_input_ids[:, :usp_chunk_size],
            hidden_states=hidden_states[:, :usp_chunk_size, :],
            position_ids=position_ids[:, : usp_chunk_size * self.sp_ulysses_degree],
            attention_mask=attention_mask[:, :usp_chunk_size],
            target_p=target_p,
            position_mask=position_mask[:, :usp_chunk_size, :],
            loss_mask=loss_mask[:, :usp_chunk_size, :],
        )

    def reduce_metrics(
        self, *, local_correct: torch.Tensor, local_denom: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        local_correct = dist_nn.all_reduce(
            local_correct, op=dist.ReduceOp.SUM, group=self.sp_group
        )
        local_denom = dist_nn.all_reduce(
            local_denom, op=dist.ReduceOp.SUM, group=self.sp_group
        )
        return local_correct, local_denom

    def reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        loss = dist_nn.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.sp_group)
        loss = loss / self.sp_world_size
        return loss
