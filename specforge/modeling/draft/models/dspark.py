"""DSpark composition with block-local Markov correction and confidence scores."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from transformers.utils import ModelOutput

from ..heads.confidence import AcceptRatePredictor
from ..heads.markov import build_markov_head
from ..hf_config import SpecForgeDraftConfig
from ..layers.inputs import ExpertRouting, LayerInputs
from ..pretrained import DraftPreTrainedModel
from .assembly import build_target_context_modules
from .context import prepare_context_inputs, run_backbone
from .registry import register_draft_model


@dataclass
class DSparkOutput(ModelOutput):
    hidden_states: Tensor | None = None
    logits: Tensor | None = None
    confidence_scores: Tensor | None = None
    expert_routing: tuple[ExpertRouting, ...] | None = None  # one per MoE layer


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
        self.backbone, self.fusion, self.embed_tokens, self.lm_head = (
            build_target_context_modules(
                draft, spec.target_features, backend=backend, device=device, dtype=dtype
            )
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
        hidden, expert_routing = run_backbone(
            self.backbone, self.embed_tokens(input_ids), prepared
        )
        if not output_logits:
            return DSparkOutput(hidden_states=hidden, expert_routing=expert_routing)
        return DSparkOutput(
            hidden_states=hidden,
            expert_routing=expert_routing,
            **self.project(hidden, predecessor_ids=predecessor_ids),
        )

    def project(self, hidden: Tensor, *, predecessor_ids: Tensor | None = None):
        """Project whole blocks so recurrent Markov state never crosses a chunk."""
        slots = self.config.draft.spec_config.protocol.num_model_slots
        if hidden.ndim != 3 or hidden.shape[1] % slots:
            raise ValueError("DSpark projection requires whole blocks")
        if self.markov_head is not None and (
            predecessor_ids is None
            or predecessor_ids.shape != hidden.shape[:2]
            or predecessor_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "configured Markov head requires integer predecessor_ids "
                "aligned with hidden states"
            )
        logits = self.lm_head(hidden)
        if self.markov_head is not None:
            shape = (hidden.shape[0], hidden.shape[1] // slots, slots)
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
        return {"logits": logits, "confidence_scores": confidence}
