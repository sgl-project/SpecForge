"""DFlash backbone with a trainable first-candidate/MTP fusion interface."""

import torch
from torch import nn

from .dflash import DFlashDraftModel
from .registry import register_draft


@register_draft
class DMTPDraftModel(DFlashDraftModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        width = config.hidden_size
        mtp_width = int(getattr(config, "target_hidden_size", width))
        self.mtp_embedding_norm = nn.RMSNorm(width, eps=config.rms_norm_eps)
        self.mtp_hidden_norm = nn.RMSNorm(mtp_width, eps=config.rms_norm_eps)
        self.mtp_fusion = nn.Linear(width + mtp_width, width, bias=False)
        nn.init.normal_(self.mtp_fusion.weight, std=config.initializer_range)

    def fuse_mtp(self, token_embedding, mtp_hidden):
        return self.mtp_fusion(
            torch.cat(
                [
                    self.mtp_embedding_norm(token_embedding),
                    self.mtp_hidden_norm(mtp_hidden.detach()),
                ],
                dim=-1,
            )
        )

    def spec_generate(self, *args, **kwargs):
        raise NotImplementedError(
            "DMTP inference requires native MTP prediction and fusion; "
            "the standard DFlash generation loop is not compatible"
        )
