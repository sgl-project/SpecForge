"""P-EAGLE composition with learned auxiliary-feature replacement in forward."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import initialization as init
from transformers.utils import ModelOutput

from ..hf_config import SpecForgeDraftConfig
from ..layers.inputs import LayerInputs
from ..pretrained import DraftPreTrainedModel
from .eagle_backbone import build_eagle_backbone
from .eagle_fusion import AuxiliaryFeatureProjection
from .registry import register_draft_model
from .tokens import build_token_modules
from .vocabulary import DraftVocabulary


@dataclass
class PEagleOutput(ModelOutput):
    hidden_states: Tensor | None = None
    logits: Tensor | None = None


@register_draft_model
class PEagleDraftModel(DraftPreTrainedModel):
    """Replace selected raw auxiliary features, then project and run the backbone.

    feature_mask=True selects the learned mask vector. Callers supply aligned
    tokens, feature-mask positions and visibility; sampling depth and scheduling
    remain outside the model. Returned hidden states are already normalized.
    """

    _no_split_modules = ["EagleDecoderStage", "DecoderLayer"]

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
            normalize_features=False,
            norm=draft.norm_config,
            device=device,
            dtype=dtype,
        )
        self.mask_hidden = nn.Parameter(
            torch.empty(spec.mask_hidden_shape, device=device, dtype=dtype)
        )
        self.embed_tokens, self.lm_head = build_token_modules(
            draft, padding_idx=spec.pad_token_id, device=device, dtype=dtype
        )
        self.vocabulary = DraftVocabulary(
            vocab_size=draft.vocab_size,
            draft_vocab_size=draft.draft_vocab_size,
            device=device,
        )
        self.post_init()

    def reset_draft_parameters(self):
        # Legacy P-EAGLE starts this vector with standard-normal entries.
        init.normal_(self.mask_hidden, mean=0, std=1)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def compute_logits(self, hidden: Tensor) -> Tensor:
        """Project the already normalized hidden states returned by forward."""
        self.vocabulary.require_ready()
        return self.lm_head(hidden)

    def forward(
        self,
        input_ids: Tensor,
        target_features: Tensor,
        layer_inputs: Sequence[LayerInputs],
        *,
        feature_mask: Tensor | None = None,
        output_logits: bool = True,
    ) -> PEagleOutput:
        if input_ids.ndim != 2 or target_features.shape != (
            *input_ids.shape,
            self.config.draft.spec_config.target_features.input_width,
        ):
            raise ValueError(
                "target features must align with input IDs and auxiliary width"
            )
        if type(output_logits) is not bool:
            raise TypeError("output_logits must be boolean")
        if feature_mask is not None:
            if (
                feature_mask.shape != input_ids.shape
                or feature_mask.dtype != torch.bool
            ):
                raise ValueError("feature_mask must be boolean [batch, sequence]")
            target_features = torch.where(
                feature_mask[..., None],
                self.mask_hidden.to(target_features.dtype),
                target_features,
            )
        features = self.fusion(target_features)
        hidden = self.backbone(
            torch.cat((self.embed_tokens(input_ids), features), dim=-1), layer_inputs
        )
        return PEagleOutput(
            hidden_states=hidden,
            logits=self.compute_logits(hidden) if output_logits else None,
        )
