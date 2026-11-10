"""Feed-forward structure, serialized as mlp_configs."""

from dataclasses import dataclass
from typing import ClassVar

from .._validation import nonempty_string, positive_int
from .base import FFNSpec

# Parameter-free activations supported by DenseFFN; modeling parity tests cover
# every entry without making configuration depend on torch or transformers.
SUPPORTED_ACTIVATIONS = frozenset(
    {"silu", "swish", "gelu", "gelu_pytorch_tanh", "relu"}
)


@dataclass(frozen=True, kw_only=True)
class DenseFFNSpec(FFNSpec):
    type: ClassVar[str] = "dense"

    intermediate_size: int
    hidden_act: str

    def __post_init__(self):
        positive_int("intermediate_size", self.intermediate_size)
        nonempty_string("hidden_act", self.hidden_act)
        if self.hidden_act not in SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"unsupported dense hidden_act {self.hidden_act!r}; "
                f"supported: {sorted(SUPPORTED_ACTIVATIONS)}"
            )
