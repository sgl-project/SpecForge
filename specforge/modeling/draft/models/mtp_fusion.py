"""Model-owned MTP fusion, independent of native checkpoint names and layers."""

from specforge.config.draft.architecture import MTPSpec
from specforge.config.draft.norm import NormSpec

from .token_feature_fusion import NormalizedTokenFeatureFusion


class MTPInputFusion(NormalizedTokenFeatureFusion):
    """Fuse already aligned token embeddings and target final hidden states.

    Token shifting and teacher execution are input-preparation responsibilities.
    This module preserves gradients for both inputs and its own parameters.
    """

    def __init__(
        self,
        config: MTPSpec,
        *,
        hidden_size: int,
        norm: NormSpec,
        device=None,
        dtype=None,
    ):
        if type(config) is not MTPSpec:
            raise TypeError("MTP input fusion requires MTPSpec")
        super().__init__(
            hidden_size=hidden_size,
            feature_size=config.target_features.hidden_size,
            norm=norm,
            device=device,
            dtype=dtype,
        )
        self.config = config
