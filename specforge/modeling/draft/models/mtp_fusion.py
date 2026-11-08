"""Model-owned MTP fusion, independent of native checkpoint names and layers."""

import torch
from torch import Tensor, nn

from specforge.config.draft.architecture import MTPSpec
from specforge.config.draft.norm import NormSpec

from ..layers import build_component


class MTPInputFusion(nn.Module):
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
        super().__init__()
        if type(config) is not MTPSpec:
            raise TypeError("MTP input fusion requires MTPSpec")
        width = config.fusion_input_width(hidden_size)
        self.config = config
        self.hidden_size = hidden_size
        self.pre_fc_norm_embedding = build_component(
            "norm", norm, hidden_size=hidden_size, device=device, dtype=dtype
        )
        self.pre_fc_norm_hidden = build_component(
            "norm",
            norm,
            hidden_size=config.target_features.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.fc = nn.Linear(width, hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, embeddings: Tensor, features: Tensor) -> Tensor:
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.hidden_size:
            raise ValueError("embeddings must be [batch, sequence, hidden_size]")
        if (
            features.ndim != 3
            or features.shape[-1] != self.config.target_features.hidden_size
            or features.shape[:2] != embeddings.shape[:2]
        ):
            raise ValueError(
                "target features must have the configured width and align with embeddings"
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
