"""DSpark composition with block-local Markov correction and confidence scores."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from transformers.utils import ModelOutput

from ..heads.confidence import AcceptRatePredictor
from ..heads.markov import build_markov_head
from ..hf_config import SpecForgeDraftConfig
from ..layers.inputs import LayerInputs
from ..pretrained import DraftPreTrainedModel
from .assembly import build_decoder_backbone
from .context import prepare_context_inputs
from .registry import register_draft_model
from .target_features import TargetFeatureFusion
from .tokens import build_token_modules


@dataclass
class DSparkOutput(ModelOutput):
    hidden_states: Tensor | None = None
    logits: Tensor | None = None
    confidence_scores: Tensor | None = None


@register_draft_model
class DSparkDraftModel(DraftPreTrainedModel):
    """Consume whole draft-only blocks and the token preceding each slot's target.

    Recurrent Markov state resets for every block. Confidence values are raw
    scores, matching the existing head; loss transforms and sampling belong to
    callers. output_logits=False skips both prediction heads.
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
        self.embed_tokens, self.lm_head = build_token_modules(
            draft, device=device, dtype=dtype
        )
        self.markov_head = build_markov_head(
            spec.markov,
            vocab_size=draft.vocab_size,
            hidden_size=draft.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.confidence_head = (
            AcceptRatePredictor(
                spec.confidence.input_width(draft.hidden_size, spec.markov),
                device=device,
                dtype=dtype,
            )
            if spec.confidence is not None
            else None
        )
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        target_features: Tensor,
        layer_inputs: Sequence[LayerInputs],
        *,
        predecessor_ids: Tensor | None = None,
        output_logits: bool = True,
    ) -> DSparkOutput:
        slots = self.config.draft.spec_config.protocol.num_model_slots
        if input_ids.ndim != 2 or input_ids.shape[1] % slots:
            raise ValueError("input_ids must be [batch, whole draft blocks]")
        if type(output_logits) is not bool:
            raise TypeError("output_logits must be boolean")
        if output_logits and self.markov_head is not None and predecessor_ids is None:
            raise ValueError("configured Markov head requires predecessor_ids")
        if predecessor_ids is not None and (
            predecessor_ids.shape != input_ids.shape
            or predecessor_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "predecessor_ids must be integer IDs aligned with input query positions"
            )
        prepared = prepare_context_inputs(
            self.fusion,
            target_features,
            layer_inputs,
            num_layers=self.backbone.num_layers,
        )
        hidden = self.backbone(self.embed_tokens(input_ids), prepared)
        if not output_logits:
            return DSparkOutput(hidden_states=hidden)
        logits = self.lm_head(hidden)
        if self.markov_head is not None:
            shape = (input_ids.shape[0], input_ids.shape[1] // slots, slots)
            logits = self.markov_head.apply_block_logits(
                logits.reshape(*shape, logits.shape[-1]),
                token_ids=predecessor_ids.reshape(shape),
                hidden_states=hidden.reshape(*shape, hidden.shape[-1]),
            ).reshape_as(logits)
        confidence = None
        if self.confidence_head is not None:
            features = hidden
            if self.config.draft.spec_config.confidence.with_markov:
                previous = self.markov_head.get_prev_embeddings(predecessor_ids).to(
                    hidden.dtype
                )
                features = torch.cat((hidden, previous), dim=-1)
            confidence = self.confidence_head(features).float()
        return DSparkOutput(
            hidden_states=hidden, logits=logits, confidence_scores=confidence
        )
