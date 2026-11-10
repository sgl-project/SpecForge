"""DFlash2 composition with sublayer convolutions and a candidate selector."""

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


@register_draft_model
class DFlash2DraftModel(DraftPreTrainedModel):
    """DFlash plus sublayer convolutions and a low-rank candidate selector.

    `logits` are the transformed unary logits. Callers pick candidates from them
    and score those through `candidate_selector` inside the same FSDP-wrapped
    forward; sampling and losses stay theirs.
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

    def unary_logits_transform_is_identity(self) -> bool:
        """Losses skip the FP32 logit copy when the transform only upcasts."""
        spec = self.config.draft.spec_config
        return spec.output_multiplier == 1 and spec.final_logit_softcapping is None

    def transform_unary_logits(self, logits: Tensor) -> Tensor:
        spec = self.config.draft.spec_config
        if self.unary_logits_transform_is_identity():
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
        output_logits: bool = True,
    ) -> DFlash2Output:
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
        logits = (
            self.transform_unary_logits(self.lm_head(hidden)) if output_logits else None
        )
        return DFlash2Output(hidden_states=hidden, logits=logits)
