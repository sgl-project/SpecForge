# coding=utf-8
"""Domino draft model entry point."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from specforge.config.draft.architecture.domino import GRUHeadSpec
from specforge.utils import get_device_type

from .dflash import DFlashDraftModel, sample
from .dflash_kernels import DFlashKernels
from .heads.domino_gru import DominoGRUHead, _correction_logits, _run_gru
from .registry import register_draft


@register_draft
class DominoDraftModel(DFlashDraftModel):
    """DFlash backbone with Domino's GRU logits correction."""

    expected_projector_type = "domino"

    def __init__(self, config, dflash_kernels: Optional[DFlashKernels] = None) -> None:
        dflash_config = getattr(config, "dflash_config", None) or {}
        projector_type = dflash_config.get("projector_type")
        if projector_type is None:
            dflash_config["projector_type"] = self.expected_projector_type
            config.dflash_config = dflash_config
        elif projector_type != self.expected_projector_type:
            raise ValueError(
                "DominoDraftModel requires " "dflash_config.projector_type='domino'."
            )
        super().__init__(config, dflash_kernels=dflash_kernels)

    def _init_draft_head(self, config, dflash_config: dict) -> None:
        self.emb_dim = int(dflash_config["emb_dim"])
        self.gru_hidden_dim = int(dflash_config["gru_hidden_dim"])
        self.pure_draft_prefix_len = int(dflash_config.get("pure_draft_prefix_len", 0))
        self.shift_label = bool(dflash_config.get("shift_label", False))

        head = DominoGRUHead(
            GRUHeadSpec(hidden_size=self.gru_hidden_dim, projection_size=self.emb_dim),
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
        )
        # Keep the legacy root-level checkpoint names without registering aliases.
        self.prefix_gru = head.prefix_gru
        self.embed_proj = head.embed_proj

    @property
    def suffix_start(self) -> int:
        return (
            self.pure_draft_prefix_len
            if self.shift_label
            else (1 + self.pure_draft_prefix_len)
        )

    def _run_gru(self, gru_inputs: torch.Tensor) -> torch.Tensor:
        return _run_gru(self.prefix_gru, gru_inputs, device_type=get_device_type())

    def _sample_draft_tokens(
        self,
        target: nn.Module,
        draft_hidden: torch.Tensor,
        block_output_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """Sample a block while causally applying Domino's GRU correction."""
        base_logits = target.lm_head(draft_hidden)
        completed_ids = block_output_ids.clone()
        base_logits4d = base_logits.unsqueeze(1)
        hidden_states4d = draft_hidden.unsqueeze(1)

        for token_position in range(1, completed_ids.shape[1]):
            previous_embeddings = target.model.embed_tokens(completed_ids).unsqueeze(1)
            final_logits = self.apply_logits_head(
                base_logits4d,
                prev_token_embeddings=previous_embeddings,
                hidden_states=hidden_states4d,
            )
            head_position = token_position - 1 if self.shift_label else token_position
            next_token_logits = final_logits[:, 0, head_position, :].unsqueeze(1)
            completed_ids[:, token_position] = sample(next_token_logits).squeeze(1)

        return completed_ids[:, 1:]

    def apply_logits_head(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
        prev_token_embeddings: Optional[torch.Tensor] = None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        del prev_token_ids
        correction_logits = self.compute_correction_logits(
            prev_token_embeddings=prev_token_embeddings,
            hidden_states=hidden_states,
        )
        prefix_logits = base_logits[:, :, : self.suffix_start, :]
        suffix_logits = base_logits[:, :, self.suffix_start :, :] + correction_logits
        return torch.cat([prefix_logits, suffix_logits], dim=2)

    def compute_correction_logits(
        self,
        *,
        prev_token_embeddings: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Return suffix-only Domino logits without materializing final logits."""
        if prev_token_embeddings is None:
            raise ValueError("DominoDraftModel requires prev_token_embeddings")

        prefix_embeddings = (
            prev_token_embeddings
            if self.shift_label
            else prev_token_embeddings[:, :, : hidden_states.shape[2] - 1, :]
        )
        return _correction_logits(
            self.prefix_gru,
            self.embed_proj,
            hidden_states[:, :, self.suffix_start :, :],
            prefix_embeddings,
            device_type=get_device_type(),
        )


__all__ = ["DominoDraftModel"]
