"""EAGLE3 composition with explicit initial-feature and recurrent-state inputs."""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from transformers.utils import ModelOutput

from ..hf_config import SpecForgeDraftConfig
from ..layers import build_component
from ..layers.attention.kv import AttentionKV
from ..layers.inputs import LayerInputs
from ..pretrained import DraftPreTrainedModel
from .eagle_backbone import build_eagle_backbone
from .eagle_fusion import AuxiliaryFeatureProjection
from .registry import register_draft_model
from .tokens import build_token_modules
from .vocabulary import DraftVocabulary


@dataclass
class Eagle3Output(ModelOutput):
    hidden_states: Tensor | None = None
    logits: Tensor | None = None
    kv: AttentionKV | None = None


@register_draft_model
class Eagle3DraftModel(DraftPreTrainedModel):
    """Project target features once, or consume an already projected recurrent state.

    hidden_states follows norm_output and can feed the next rollout. Logits are
    normalized exactly once and use draft-vocabulary order. Token alignment,
    rollout scheduling, masks and positions are supplied by the caller. kv_prefix
    contains projected, rotated keys and values from previous depths; returned kv
    contains only this depth and stays attached to autograd. Position inputs cover
    only new tokens, while attention visibility covers prefix plus new tokens.
    """

    _no_split_modules = ["EagleDecoderStage"]

    def __init__(
        self, config: SpecForgeDraftConfig, *, backend="eager", device=None, dtype=None
    ):
        super().__init__(config)
        draft, spec = config.draft, config.draft.spec_config
        self.backbone = build_eagle_backbone(
            draft, backend=backend, device=device, dtype=dtype
        )
        self.fusion = AuxiliaryFeatureProjection(
            spec.target_features,
            hidden_size=draft.hidden_size,
            normalize_features=spec.normalize_features,
            norm=draft.norm_config,
            device=device,
            dtype=dtype,
        )
        self.embed_tokens, self.lm_head = build_token_modules(
            draft, device=device, dtype=dtype
        )
        self.logit_norm = (
            nn.Identity()
            if spec.norm_output
            else build_component(
                "norm",
                draft.norm_config,
                hidden_size=draft.hidden_size,
                device=device,
                dtype=dtype,
            )
        )
        self.vocabulary = DraftVocabulary(
            vocab_size=draft.vocab_size,
            draft_vocab_size=draft.draft_vocab_size,
            device=device,
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def compute_logits(self, hidden: Tensor) -> Tensor:
        self.vocabulary.require_ready()
        return self.lm_head(self.logit_norm(hidden))

    def forward(
        self,
        input_ids: Tensor,
        target_features: Tensor | None,
        layer_inputs: Sequence[LayerInputs],
        *,
        recurrent_hidden: Tensor | None = None,
        kv_prefix: AttentionKV | None = None,
        output_logits: bool = True,
    ) -> Eagle3Output:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch, sequence]")
        if type(output_logits) is not bool:
            raise TypeError("output_logits must be boolean")
        if (target_features is None) == (recurrent_hidden is None):
            raise ValueError(
                "provide exactly one of target_features and recurrent_hidden"
            )
        features = (
            self.fusion(target_features)
            if target_features is not None
            else recurrent_hidden
        )
        if features.shape != (*input_ids.shape, self.config.draft.hidden_size):
            raise ValueError(
                "features must align with input IDs and draft hidden width"
            )
        # EAGLE3 has one stage. The caller owns depth iteration and K/V lifetime.
        state = self.backbone.begin(
            torch.cat((self.embed_tokens(input_ids), features), dim=-1), layer_inputs
        )
        stage = self.backbone.layers[0]
        options = dict(kv_prefix=kv_prefix, output_kv=True)
        if (
            self.backbone.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            hidden, kv = checkpoint(
                stage,
                state.hidden,
                state.layer_inputs[0],
                use_reentrant=False,
                **options,
            )
        else:
            hidden, kv = stage(state.hidden, state.layer_inputs[0], **options)
        hidden = self.backbone.finalize(replace(state, hidden=hidden, next_layer=1))
        return Eagle3Output(
            hidden_states=hidden,
            kv=kv,
            logits=self.compute_logits(hidden) if output_logits else None,
        )
