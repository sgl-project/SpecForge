"""Unfused gated dense FFN, constructed from its node and input width."""

from torch import Tensor, nn
from transformers.activations import ACT2FN

from specforge.config.draft._validation import positive_int
from specforge.config.draft.mlp import DenseFFNSpec

from ..registry import register_component_builder

# Parameter-free activation primitives with numerical parity coverage. Expanding
# this set requires a reference test; it does not add another config default.
SUPPORTED_ACTIVATIONS = frozenset(
    {"silu", "swish", "gelu", "gelu_pytorch_tanh", "relu"}
)


class DenseFFN(nn.Module):
    def __init__(
        self, config: DenseFFNSpec, *, hidden_size: int, device=None, dtype=None
    ):
        super().__init__()
        if type(config) is not DenseFFNSpec:
            raise TypeError("DenseFFN requires DenseFFNSpec")
        positive_int("hidden_size", hidden_size)
        if config.hidden_act not in SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"unsupported dense hidden_act {config.hidden_act!r}; "
                f"supported: {sorted(SUPPORTED_ACTIVATIONS)}"
            )
        self.config = config
        self.act_fn = ACT2FN[config.hidden_act]
        self.gate_proj = nn.Linear(
            hidden_size,
            config.intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.up_proj = nn.Linear(
            hidden_size,
            config.intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden)) * self.up_proj(hidden)
        )


register_component_builder("mlp", DenseFFNSpec, DenseFFN)
