"""DFlash2 composition with sublayer convolutions and explicit candidate scoring."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from transformers.utils import ModelOutput

from ..heads.candidate_selector import CandidateSelector
from ..hf_config import SpecForgeDraftConfig
from ..layers.inputs import LayerInputs
from ..pretrained import DraftPreTrainedModel
from .assembly import build_decoder_backbone
from .context import prepare_context_inputs
from .registry import register_draft_model
from .target_features import TargetFeatureFusion
from .tokens import build_token_modules


@dataclass
class DFlash2Output(ModelOutput):
    hidden_states: Tensor | None = None
    logits: Tensor | None = None
    candidate_scores: Tensor | None = None


@register_draft_model
class DFlash2DraftModel(DraftPreTrainedModel):
    """Own DFlash2's composition without inheriting another algorithm model.

    Optional candidate/predecessor IDs align with the input query positions.
    Forward scores those candidates without detaching hidden states or logits.
    Candidate construction, prediction-position masks, loss weights and sampling
    remain caller policy; the selector is also available for lattice scoring.
    """

    _no_split_modules = ["DecoderLayer", "CandidateSelector"]

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
        self.candidate_selector = CandidateSelector(
            hidden_size=draft.hidden_size,
            vocab_size=draft.vocab_size,
            state_rank=spec.selector_rank,
            top_k=spec.selector_top_k,
            initializer_range=draft.initializer_range,
            device=device,
            dtype=dtype,
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def transform_unary_logits(self, logits: Tensor) -> Tensor:
        spec = self.config.draft.spec_config
        if spec.output_multiplier == 1 and spec.final_logit_softcapping is None:
            return logits.float()
        logits = logits.float() * spec.output_multiplier
        if spec.final_logit_softcapping is not None:
            cap = spec.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        return logits

    def forward(
        self,
        input_ids: Tensor,
        target_features: Tensor,
        layer_inputs: Sequence[LayerInputs],
        *,
        candidate_ids: Tensor | None = None,
        predecessor_ids: Tensor | None = None,
        output_logits: bool = True,
    ) -> DFlash2Output:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch, draft_sequence]")
        if type(output_logits) is not bool:
            raise TypeError("output_logits must be boolean")
        if (candidate_ids is None) != (predecessor_ids is None):
            raise ValueError(
                "candidate_ids and predecessor_ids must be supplied together"
            )
        if candidate_ids is not None:
            if not output_logits:
                raise ValueError("candidate scoring requires output_logits")
            if (
                candidate_ids.ndim != 3
                or candidate_ids.shape[:2] != input_ids.shape
                or candidate_ids.shape[-1] == 0
                or predecessor_ids.shape != input_ids.shape
            ):
                raise ValueError(
                    "candidate and predecessor IDs must align with input query positions"
                )
            if candidate_ids.dtype != torch.long or predecessor_ids.dtype != torch.long:
                raise TypeError("candidate and predecessor IDs must be int64")
        prepared = prepare_context_inputs(
            self.fusion,
            target_features,
            layer_inputs,
            num_layers=self.backbone.num_layers,
        )
        hidden = self.backbone(self.embed_tokens(input_ids), prepared)
        logits = (
            self.transform_unary_logits(self.lm_head(hidden)) if output_logits else None
        )
        scores = None
        if candidate_ids is not None:
            scores = self.candidate_selector.score_candidates(
                candidate_ids=candidate_ids,
                unary_logits=logits.gather(-1, candidate_ids),
                hidden_states=hidden,
                predecessor_ids=predecessor_ids,
            )
        return DFlash2Output(
            hidden_states=hidden, logits=logits, candidate_scores=scores
        )
