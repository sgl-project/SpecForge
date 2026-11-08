"""Domino composition with aligned, block-local GRU correction."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from transformers.utils import ModelOutput

from ..heads.domino_gru import DominoGRUHead
from ..hf_config import SpecForgeDraftConfig
from ..layers.inputs import LayerInputs
from ..pretrained import DraftPreTrainedModel
from .assembly import build_decoder_backbone
from .context import prepare_context_inputs
from .registry import register_draft_model
from .target_features import TargetFeatureFusion
from .tokens import build_token_modules


@dataclass
class DominoOutput(ModelOutput):
    hidden_states: Tensor | None = None
    logits: Tensor | None = None
    correction_logits: Tensor | None = None


@register_draft_model
class DominoDraftModel(DraftPreTrainedModel):
    """Consume whole blocks and teacher/committed tokens in block slot order.

    The configured label offset aligns the GRU prefix with decoder suffix states.
    GRU state resets for every block; sampling and loss construction belong to
    callers. Correction logits retain [batch, blocks, suffix, vocab] dimensions.
    output_logits=False skips both the vocabulary projection and GRU head.
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
            target_hidden_size=draft.hidden_size,
            hidden_size=draft.hidden_size,
            norm=draft.norm_config,
            device=device,
            dtype=dtype,
        )
        self.embed_tokens, self.lm_head = build_token_modules(
            draft, device=device, dtype=dtype
        )
        self.gru_head = DominoGRUHead(
            spec.gru,
            hidden_size=draft.hidden_size,
            vocab_size=draft.vocab_size,
            device=device,
            dtype=dtype,
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
        prev_token_ids: Tensor | None = None,
        output_logits: bool = True,
    ) -> DominoOutput:
        spec = self.config.draft.spec_config
        slots = spec.protocol.num_model_slots
        if input_ids.ndim != 2 or input_ids.shape[1] % slots:
            raise ValueError("input_ids must be [batch, whole draft blocks]")
        if type(output_logits) is not bool:
            raise TypeError("output_logits must be boolean")
        if output_logits and prev_token_ids is None:
            raise ValueError("Domino GRU head requires prev_token_ids")
        if prev_token_ids is not None and (
            prev_token_ids.shape != input_ids.shape
            or prev_token_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "prev_token_ids must be integer IDs in input block slot order"
            )
        prepared = prepare_context_inputs(
            self.fusion,
            target_features,
            layer_inputs,
            num_layers=self.backbone.num_layers,
        )
        hidden = self.backbone(self.embed_tokens(input_ids), prepared)
        if not output_logits:
            return DominoOutput(hidden_states=hidden)
        shape = (input_ids.shape[0], input_ids.shape[1] // slots, slots)
        states = hidden.reshape(*shape, hidden.shape[-1])
        previous = self.embed_tokens(prev_token_ids).reshape_as(states)
        prefix = previous if spec.block.shift_label else previous[..., :-1, :]
        correction = self.gru_head(states[..., spec.suffix_start :, :], prefix)
        base = self.lm_head(hidden)
        blocks = base.reshape(*shape, base.shape[-1])
        logits = torch.cat(
            (
                blocks[..., : spec.suffix_start, :],
                blocks[..., spec.suffix_start :, :] + correction,
            ),
            dim=-2,
        ).reshape_as(base)
        return DominoOutput(
            hidden_states=hidden, logits=logits, correction_logits=correction
        )
