"""Teacher-forced DMTP: MTP predicts T1; DFlash learns T2 onward."""

import torch

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.training.strategies.base import DFlashTrainStrategy


class OnlineDMTPModel(OnlineDFlashModel):
    # Anchor positions refer to T1. Target context ends BEFORE A = T1 - 1.
    context_offset = -1

    def _sample_anchor_positions(
        self, seq_len, loss_mask, device, max_valid_anchors=None
    ):
        # Positions 0/1 cannot have both a previous target state and an A.
        eligible = loss_mask.clone()
        eligible[:, :2] = 0
        return super()._sample_anchor_positions(
            seq_len, eligible, device, max_valid_anchors=max_valid_anchors
        )

    def _create_noise_embed(
        self, input_ids, anchor_positions, block_keep_mask, *, mtp_hidden
    ):
        embeddings = super()._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )
        batch, blocks = anchor_positions.shape
        indices = (anchor_positions - 1).clamp(min=0)
        predictors = mtp_hidden.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, mtp_hidden.shape[-1])
        )
        embeddings = embeddings.reshape(batch, blocks, self.block_size, -1)
        fused = self.draft_model.fuse_mtp(embeddings[:, :, 0], predictors)
        first = torch.where(block_keep_mask.unsqueeze(-1), fused, embeddings[:, :, 0])
        return torch.cat([first.unsqueeze(2), embeddings[:, :, 1:]], dim=2).flatten(
            1, 2
        )

    def forward(self, input_ids, hidden_states, loss_mask, *, mtp_hidden, **kwargs):
        if mtp_hidden.ndim != 3 or mtp_hidden.shape[:2] != input_ids.shape:
            raise ValueError("mtp_hidden must have shape [batch, sequence, hidden]")
        if input_ids.shape[1] < 4:
            raise ValueError("DMTP requires target history, A, T1, and at least T2")
        return super().forward(
            input_ids,
            hidden_states.detach(),
            loss_mask,
            mtp_hidden=mtp_hidden.detach(),
            **kwargs,
        )


class DMTPTrainStrategy(DFlashTrainStrategy):
    name = "dmtp"
    required_features = DFlashTrainStrategy.required_features | {"mtp_hidden"}
