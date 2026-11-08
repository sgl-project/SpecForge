"""Resolved auxiliary features and concatenated-input dimensions for EAGLE."""

from dataclasses import dataclass

from .._validation import positive_int


@dataclass(frozen=True, kw_only=True)
class AuxiliaryFeatureSpec:
    target_layer_ids: tuple[int, ...]
    hidden_size: int

    def __post_init__(self):
        if type(self.target_layer_ids) is not tuple or len(self.target_layer_ids) != 3:
            raise ValueError("target_layer_ids must be a tuple of exactly three layers")
        if any(type(layer) is not int or layer < 0 for layer in self.target_layer_ids):
            raise ValueError("target_layer_ids must contain nonnegative integers")
        positive_int("target_features.hidden_size", self.hidden_size)

    @property
    def input_width(self) -> int:
        return len(self.target_layer_ids) * self.hidden_size


def attention_input_width(hidden_size: int, layer_index: int) -> int:
    """Fusion doubles the first attention's input; its output stays hidden_size."""
    positive_int("hidden_size", hidden_size)
    if type(layer_index) is not int or layer_index < 0:
        raise ValueError("layer_index must be a nonnegative integer")
    return hidden_size * (2 if layer_index == 0 else 1)


def validate_auxiliary_fusion(config) -> None:
    if "single_stream" not in config.residual_config.capabilities:
        raise ValueError("auxiliary fusion requires single_stream residuals")
    first = config.layer_configs[config.layer_types[0]]
    if (
        "variable_input_width"
        not in config.attention_configs[first.attention].capabilities
    ):
        raise ValueError("first attention must support variable_input_width for fusion")
    for name, attention in config.attention_configs.items():
        if getattr(attention, "is_causal", None) is not True:
            raise ValueError(
                f"attention_configs.{name}: EAGLE requires causal attention"
            )
