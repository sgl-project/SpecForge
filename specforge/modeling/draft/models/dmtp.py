"""DMTP fuses the external first candidate before an ordinary decoder stack."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from transformers.utils import ModelOutput

from ..hf_config import SpecForgeDraftConfig
from ..layers.inputs import LayerInputs
from ..pretrained import DraftPreTrainedModel
from .assembly import build_decoder_backbone
from .context import prepare_context_inputs
from .registry import register_draft_model
from .target_features import TargetFeatureFusion
from .token_feature_fusion import NormalizedTokenFeatureFusion
from .tokens import build_token_modules


@dataclass
class DMTPOutput(ModelOutput):
    hidden_states: Tensor | None = None
    logits: Tensor | None = None


@register_draft_model
class DMTPDraftModel(DraftPreTrainedModel):
    """Consume [T1, masked suffix] blocks and one aligned MTP feature per block.

    The caller supplies post-normalization MTP features at A, prepares target
    context ending before A, and supplies first candidates and visibility. The
    model owns learned fusion in both training and inference. It runs neither
    the frozen predictor nor candidate sampling, and never detaches its inputs.
    """

    _no_split_modules = ["DecoderLayer"]

    def __init__(
        self, config: SpecForgeDraftConfig, *, backend="eager", device=None, dtype=None
    ):
        super().__init__(config)
        draft, spec = config.draft, config.draft.spec_config
        self.backbone = build_decoder_backbone(
            draft, backend=backend, device=device, dtype=dtype
        )
        self.fusion = TargetFeatureFusion(
            spec.target_features,
            hidden_size=draft.hidden_size,
            norm=draft.norm_config,
            device=device,
            dtype=dtype,
        )
        self.mtp_fusion = NormalizedTokenFeatureFusion(
            hidden_size=draft.hidden_size,
            feature_size=spec.mtp_features.hidden_size,
            norm=draft.norm_config,
            device=device,
            dtype=dtype,
        )
        self.embed_tokens, self.lm_head = build_token_modules(
            draft, device=device, dtype=dtype
        )
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        target_features: Tensor,
        layer_inputs: Sequence[LayerInputs],
        *,
        mtp_features: Tensor,
        block_keep_mask: Tensor | None = None,
        output_logits: bool = True,
    ) -> DMTPOutput:
        slots = self.config.draft.spec_config.protocol.num_model_slots
        if input_ids.ndim != 2 or input_ids.shape[1] % slots:
            raise ValueError("input_ids must be [batch, whole DMTP blocks]")
        if type(output_logits) is not bool:
            raise TypeError("output_logits must be boolean")
        shape = (input_ids.shape[0], input_ids.shape[1] // slots)
        if block_keep_mask is not None and (
            block_keep_mask.shape != shape or block_keep_mask.dtype != torch.bool
        ):
            raise ValueError("block_keep_mask must be boolean [batch, blocks]")
        if mtp_features.ndim != 3 or mtp_features.shape[:2] != shape:
            raise ValueError("mtp_features must be [batch, blocks, feature_size]")
        if block_keep_mask is not None:
            # Mask before fusion so padded NaNs cannot contaminate parameter gradients.
            mtp_features = mtp_features.masked_fill(~block_keep_mask[..., None], 0)
        embeddings = self.embed_tokens(input_ids).reshape(
            *shape, slots, self.config.draft.hidden_size
        )
        first = self.mtp_fusion(embeddings[..., 0, :], mtp_features)
        if block_keep_mask is not None:
            first = torch.where(
                block_keep_mask[..., None], first, embeddings[..., 0, :]
            )
        assembled = torch.cat(
            (first.unsqueeze(-2), embeddings[..., 1:, :]), dim=-2
        ).flatten(1, 2)
        prepared = prepare_context_inputs(
            self.fusion,
            target_features,
            layer_inputs,
            num_layers=self.backbone.num_layers,
        )
        hidden = self.backbone(assembled, prepared)
        return DMTPOutput(
            hidden_states=hidden,
            logits=self.compute_logits(hidden) if output_logits else None,
        )
