"""Canonical DFlash: token modules, target fusion and a sequential backbone."""

from collections.abc import Sequence
from dataclasses import dataclass, replace

from torch import Tensor, nn
from transformers.utils import ModelOutput

from ..bindings import BoundEmbedding, BoundLMHead
from ..hf_config import SpecForgeDraftConfig
from ..layers.inputs import LayerInputs
from ..pretrained import DraftPreTrainedModel
from .assembly import build_decoder_backbone
from .registry import register_draft_model
from .target_features import TargetFeatureFusion


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
        # The current DFlash feature contract uses the draft hidden width for
        # each selected target layer, matching the canonical/legacy codec.
        self.fusion = TargetFeatureFusion(
            draft.spec_config.target_features,
            target_hidden_size=draft.hidden_size,
            hidden_size=draft.hidden_size,
            norm=draft.norm_config,
            device=device,
            dtype=dtype,
        )
        options = dict(device=device, dtype=dtype)
        self.embed_tokens = (
            BoundEmbedding(draft.vocab_size, draft.hidden_size, **options)
            if "embed_tokens" in draft.target_bindings
            else nn.Embedding(draft.vocab_size, draft.hidden_size, **options)
        )
        self.lm_head = (
            BoundLMHead(
                draft.target_logit_transform,
                vocab_size=draft.vocab_size,
                hidden_size=draft.hidden_size,
                **options,
            )
            if "lm_head" in draft.target_bindings
            else nn.Linear(
                draft.hidden_size, draft.draft_vocab_size, bias=False, **options
            )
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
        if len(layer_inputs) != self.backbone.num_layers:
            raise ValueError("layer_inputs must contain one entry per layer")
        for inputs in layer_inputs:
            if not isinstance(inputs, LayerInputs):
                raise TypeError("each layer input must be LayerInputs")
            if inputs.target_context is not None:
                raise ValueError(
                    "DFlash computes target_context through its trainable fusion"
                )
        context = self.fusion(target_features)
        hidden = self.backbone(
            self.embed_tokens(input_ids),
            [replace(inputs, target_context=context) for inputs in layer_inputs],
        )
        return DFlashOutput(
            hidden_states=hidden,
            logits=self.lm_head(hidden) if output_logits else None,
        )
