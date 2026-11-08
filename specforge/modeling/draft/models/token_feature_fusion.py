"""Normalized token/feature fusion shared by MTP and first-candidate DMTP inputs."""

import torch
from torch import Tensor, nn

from specforge.config.draft._validation import positive_int
from specforge.config.draft.norm import NormSpec

from ..layers import build_component


class NormalizedTokenFeatureFusion(nn.Module):
    """Fuse aligned [batch, positions, width] tensors without shifting or detaching.

    The model supplies dimensions and chooses which slots use this computation.
    Teacher execution and freezing remain the feature provider's responsibility.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        feature_size: int,
        norm: NormSpec,
        device=None,
        dtype=None,
    ):
        super().__init__()
        positive_int("hidden_size", hidden_size)
        positive_int("feature_size", feature_size)
        self.hidden_size, self.feature_size = hidden_size, feature_size
        self.pre_fc_norm_embedding = build_component(
            "norm", norm, hidden_size=hidden_size, device=device, dtype=dtype
        )
        self.pre_fc_norm_hidden = build_component(
            "norm", norm, hidden_size=feature_size, device=device, dtype=dtype
        )
        self.fc = nn.Linear(
            hidden_size + feature_size,
            hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, embeddings: Tensor, features: Tensor) -> Tensor:
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.hidden_size:
            raise ValueError("embeddings must be [batch, positions, hidden_size]")
        if (
            features.ndim != 3
            or features.shape[-1] != self.feature_size
            or features.shape[:2] != embeddings.shape[:2]
        ):
            raise ValueError(
                "features must have the configured width and align with embeddings"
            )
        return self.fc(
            torch.cat(
                (
                    self.pre_fc_norm_embedding(embeddings),
                    self.pre_fc_norm_hidden(features),
                ),
                dim=-1,
            )
        )
