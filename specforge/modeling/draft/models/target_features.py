"""Model-owned target-context fusion shared by DFlash-family architectures."""

from torch import Tensor, nn

from specforge.config.draft._validation import positive_int
from specforge.config.draft.architecture.features import TargetFeatureSpec
from specforge.config.draft.norm import NormSpec

from ..layers import build_component, component_builder


class TargetFeatureFusion(nn.Module):
    """Project concatenated target features, then normalize the context stream.

    The caller selects and concatenates features in target_layer_ids order.
    Target width is explicit; it is not inferred from the draft's hidden size.
    No feature selection, masking, teacher execution or detaching happens here.
    """

    def __init__(
        self,
        config: TargetFeatureSpec,
        *,
        hidden_size: int,
        norm: NormSpec,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if type(config) is not TargetFeatureSpec:
            raise TypeError("target fusion requires TargetFeatureSpec")
        positive_int("hidden_size", hidden_size)
        self.input_size = config.input_width()
        component_builder("norm", norm.type)
        self.config = config
        self.fc = nn.Linear(
            self.input_size, hidden_size, bias=False, device=device, dtype=dtype
        )
        self.hidden_norm = build_component(
            "norm", norm, hidden_size=hidden_size, device=device, dtype=dtype
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[-1] != self.input_size:
            raise ValueError(
                "target features must be [batch, context, concatenated_width]"
            )
        return self.hidden_norm(self.fc(features))
