"""References composing an ordinary decoder layer, with no type dispatch."""

from dataclasses import dataclass
from types import MappingProxyType

from ._validation import nonempty_string


@dataclass(frozen=True, kw_only=True)
class LayerSpec:
    attention: str
    mlp: str
    sublayer_conv: str | None = None

    def __post_init__(self):
        nonempty_string("attention", self.attention)
        nonempty_string("mlp", self.mlp)
        if self.sublayer_conv is not None:
            nonempty_string("sublayer_conv", self.sublayer_conv)


KIND_BY_REFERENCE_FIELD = MappingProxyType(
    {"attention": "attention", "mlp": "mlp", "sublayer_conv": "conv"}
)
