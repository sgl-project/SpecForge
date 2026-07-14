from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class StepState:
    input_ids: torch.Tensor
    hidden_states: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_p: torch.Tensor
    target_p_on_draft: torch.Tensor
    target_token_ids: torch.Tensor
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
            input_ids=global_input_ids,
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            target_p=target_p,
            target_p_on_draft=target_p_on_draft,
            target_token_ids=target_token_ids,
            position_mask=position_mask,
            loss_mask=loss_mask,
        )
