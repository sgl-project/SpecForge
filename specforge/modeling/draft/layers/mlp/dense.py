"""Gated dense FFN with an optional fused activation and unchanged parameters."""

from torch import Tensor, nn
from transformers.activations import ACT2FN

from specforge.config.draft._validation import positive_int
from specforge.config.draft.mlp import DenseFFNSpec

from ..registry import register_component_builder


class DenseFFN(nn.Module):
    def __init__(
        self, config: DenseFFNSpec, *, hidden_size: int, device=None, dtype=None
    ):
        super().__init__()
        if type(config) is not DenseFFNSpec:
            raise TypeError("DenseFFN requires DenseFFNSpec")
        positive_int("hidden_size", hidden_size)
        self.config = config
        self.use_liger_kernel = False
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
        if self.use_liger_kernel:
            from liger_kernel.ops.swiglu import LigerSiLUMulFunction

            return self.down_proj(
                LigerSiLUMulFunction.apply(self.gate_proj(hidden), self.up_proj(hidden))
            )
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden)) * self.up_proj(hidden)
        )


register_component_builder("mlp", DenseFFNSpec, DenseFFN)
