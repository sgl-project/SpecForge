"""Canonical MTP composition, independent of target-family implementations."""

from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor
from transformers.utils import ModelOutput

from ..hf_config import SpecForgeDraftConfig
from ..layers.inputs import LayerInputs
from ..pretrained import DraftPreTrainedModel
from .assembly import build_decoder_backbone
from .mtp_fusion import MTPInputFusion
from .registry import register_draft_model
from .tokens import build_token_modules


@dataclass
class MTPOutput(ModelOutput):
    hidden_states: Tensor | None = None
    logits: Tensor | None = None


@register_draft_model
class MTPDraftModel(DraftPreTrainedModel):
    """Fuse aligned final target features and token embeddings, then decode.

    At input position t, the caller supplies the target feature from t-1;
    the output predicts t+1. MTPSpec owns these offsets. This forward neither
    shifts tensors nor runs the teacher, and has no native-family selection.
    """

    _no_split_modules = ["DecoderLayer"]

    def __init__(
        self, config: SpecForgeDraftConfig, *, backend="eager", device=None, dtype=None
    ):
        super().__init__(config)
        draft = config.draft
        self.backbone = build_decoder_backbone(
            draft, backend=backend, device=device, dtype=dtype
        )
        self.fusion = MTPInputFusion(
            draft.spec_config,
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
    ) -> MTPOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch, sequence]")
        if type(output_logits) is not bool:
            raise TypeError("output_logits must be boolean")
        if len(layer_inputs) != self.backbone.num_layers:
            raise ValueError("layer_inputs must contain one entry per layer")
        for inputs in layer_inputs:
            if not isinstance(inputs, LayerInputs):
                raise TypeError("each layer input must be LayerInputs")
            if inputs.target_context is not None:
                raise ValueError(
                    "MTP consumes features through input fusion, not target KV context"
                )
        hidden = self.backbone(
            self.fusion(self.embed_tokens(input_ids), target_features), layer_inputs
        )
        return MTPOutput(
            hidden_states=hidden, logits=self.lm_head(hidden) if output_logits else None
        )
