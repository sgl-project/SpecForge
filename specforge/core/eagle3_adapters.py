from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn

from specforge.distributed import get_draft_sp_group, get_sp_ulysses_group


@dataclass
class BackboneStepState:
    input_ids: torch.Tensor
    hidden_states: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor


@dataclass
class StepState(BackboneStepState):
    target_p: torch.Tensor
    target_p_on_draft: torch.Tensor
    target_token_ids: torch.Tensor
    position_mask: torch.Tensor
    loss_mask: torch.Tensor


class BackendAdapter:
    def __init__(self, model: "OnlineEagle3Model"):
        self.m = model

    def backbone_row_count(self, *, seq_length: int, ttt_length: int) -> int:
        return seq_length

    def backbone_view(
        self,
        *,
        row_count: int,
        global_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> BackboneStepState:
        if row_count != hidden_states.shape[1]:
            raise ValueError(
                f"backbone row_count ({row_count}) must equal local hidden-state "
                f"length ({hidden_states.shape[1]})"
            )
        return BackboneStepState(
            input_ids=global_input_ids,
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

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
        target_p_on_draft_padded: Optional[torch.Tensor] = None,
        target_token_ids_padded: Optional[torch.Tensor] = None,
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
        target_p_on_draft_padded: Optional[torch.Tensor] = None,
        target_token_ids_padded: Optional[torch.Tensor] = None,
        position_mask: torch.Tensor,
        seq_length: int,
    ) -> StepState:
        backbone = self.backbone_view(
            row_count=seq_length,
            global_input_ids=global_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            hidden_states=hidden_states,
        )
        if target_p_on_draft_padded is None:
            target_p_on_draft_padded = target_p_padded
        if target_token_ids_padded is None:
            target_token_ids_padded = target_p_padded.argmax(dim=-1)
        target_p = target_p_padded[:, idx : idx + seq_length, :].contiguous()
        target_p_on_draft = target_p_on_draft_padded[
            :, idx : idx + seq_length, :
        ].contiguous()
        target_token_ids = target_token_ids_padded[
            :, idx : idx + seq_length
        ].contiguous()
        return StepState(
            input_ids=backbone.input_ids,
            hidden_states=backbone.hidden_states,
            position_ids=backbone.position_ids,
            attention_mask=backbone.attention_mask,
            target_p=target_p,
            target_p_on_draft=target_p_on_draft,
            target_token_ids=target_token_ids,
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

    def backbone_row_count(self, *, seq_length: int, ttt_length: int) -> int:
        row_count = seq_length - ttt_length
        if row_count <= 0:
            raise ValueError(
                f"USP local seq_length ({seq_length}) must be larger than "
                f"ttt_length ({ttt_length})"
            )
        return row_count

    def backbone_view(
        self,
        *,
        row_count: int,
        global_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> BackboneStepState:
        return BackboneStepState(
            input_ids=global_input_ids[:, :row_count],
            hidden_states=hidden_states[:, :row_count, :],
            position_ids=position_ids[:, : row_count * self.sp_ulysses_degree],
            attention_mask=attention_mask[:, :row_count],
        )

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
        target_p_on_draft_padded: Optional[torch.Tensor] = None,
        target_token_ids_padded: Optional[torch.Tensor] = None,
        position_mask: torch.Tensor,
        seq_length: int,
    ) -> StepState:
        if target_p_on_draft_padded is None:
            target_p_on_draft_padded = target_p_padded
        if target_token_ids_padded is None:
            target_token_ids_padded = target_p_padded.argmax(dim=-1)
        usp_chunk_size = self.backbone_row_count(
            seq_length=seq_length, ttt_length=ttt_length
        )
        backbone = self.backbone_view(
            row_count=usp_chunk_size,
            global_input_ids=global_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            hidden_states=hidden_states,
        )
        target_p = target_p_padded[:, idx : idx + usp_chunk_size, :]
        target_p_on_draft = target_p_on_draft_padded[:, idx : idx + usp_chunk_size, :]
        target_token_ids = target_token_ids_padded[:, idx : idx + usp_chunk_size]
        return StepState(
            input_ids=backbone.input_ids,
            hidden_states=backbone.hidden_states,
            position_ids=backbone.position_ids,
            attention_mask=backbone.attention_mask,
            target_p=target_p,
            target_p_on_draft=target_p_on_draft,
            target_token_ids=target_token_ids,
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
        return loss
