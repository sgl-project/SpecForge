"""Canonical DFlash: token modules, target fusion and a sequential backbone."""

from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor
from transformers.utils import ModelOutput

from ..hf_config import SpecForgeDraftConfig
from ..layers.inputs import ExpertChoice, LayerInputs, MoELayerOutput
from ..pretrained import DraftPreTrainedModel
from .assembly import build_target_context_modules
from .context import prepare_context_inputs
from .registry import register_draft_model


@dataclass
class DFlashOutput(ModelOutput):
    hidden_states: Tensor | None = None
    logits: Tensor | None = None
    expert_choices: tuple[ExpertChoice, ...] | None = None  # one per MoE layer


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
        self.backbone, self.fusion, self.embed_tokens, self.lm_head = (
            build_target_context_modules(
                draft,
                draft.spec_config.target_features,
                backend=backend,
                device=device,
                dtype=dtype,
            )
        )
        self.post_init()

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
        hidden, outputs = self.backbone.forward_with_layer_outputs(
            self.embed_tokens(input_ids), prepared
        )
        choices = tuple(
            o.expert_choice for o in outputs if isinstance(o, MoELayerOutput)
        )
        return DFlashOutput(
            hidden_states=hidden,
            logits=self.compute_logits(hidden) if output_logits else None,
            expert_choices=choices or None,
        )
