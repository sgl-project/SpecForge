"""Canonical DFlash: token modules, target fusion and a sequential backbone."""

from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor
from transformers.utils import ModelOutput

from ..hf_config import SpecForgeDraftConfig
from ..layers.inputs import LayerInputs
from ..pretrained import DraftPreTrainedModel
from .assembly import build_decoder_backbone
from .context import prepare_context_inputs
from .registry import register_draft_model
from .target_features import TargetFeatureFusion
from .tokens import build_token_modules


@dataclass
class DFlashOutput(ModelOutput):
    hidden_states: Tensor | None = None
    logits: Tensor | None = None


@register_draft_model
class DFlashDraftModel(DraftPreTrainedModel):
    """In-memory model construction; loaders install any target-bound weights.

    Inputs contain aligned target features and explicit per-layer attention
    layouts/positions. Sampling, loss computation and mutable serving caches
    belong to their callers. The legacy DFlash entry point remains independent.
    """

    _no_split_modules = ["DecoderLayer"]

    def __init__(
        self,
        config: SpecForgeDraftConfig,
        *,
        backend: str = "eager",
        device=None,
        dtype=None,
    ):
        super().__init__(config)
        draft = config.draft
        self.backbone = build_decoder_backbone(
            draft, backend=backend, device=device, dtype=dtype
        )
        self.fusion = TargetFeatureFusion(
            draft.spec_config.target_features,
            hidden_size=draft.hidden_size,
            norm=draft.norm_config,
            device=device,
            dtype=dtype,
        )
        self.embed_tokens, self.lm_head = build_token_modules(
            draft, device=device, dtype=dtype
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids: Tensor,
        target_features: Tensor,
        layer_inputs: Sequence[LayerInputs],
        *,
        output_logits: bool = True,
    ) -> DFlashOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch, draft_sequence]")
        if type(output_logits) is not bool:
            raise TypeError("output_logits must be boolean")
        prepared = prepare_context_inputs(
            self.fusion,
            target_features,
            layer_inputs,
            num_layers=self.backbone.num_layers,
        )
        hidden = self.backbone(self.embed_tokens(input_ids), prepared)
        return DFlashOutput(
            hidden_states=hidden,
            logits=self.lm_head(hidden) if output_logits else None,
        )
