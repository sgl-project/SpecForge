"""Compose pre-norm attention/FFN sublayers with single-stream residuals."""

from torch import Tensor, nn

from specforge.config.draft._validation import positive_int
from specforge.config.draft.norm import NormSpec

from .inputs import LayerInputs
from .registry import build_component


class DecoderLayer(nn.Module):
    """Standard residual wiring shared by model-owned and sequential execution.

    The model supplies constructed attention/FFN components. Input fusion,
    inter-layer injection, multiple residual streams, and file loading are not
    implemented here. Sublayer convolution requires its own implementation and
    is rejected until its execution contract is available.
    """

    def __init__(
        self,
        attention: nn.Module,
        mlp: nn.Module,
        *,
        hidden_size: int,
        norm: NormSpec,
        sublayer_conv: nn.Module | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        positive_int("hidden_size", hidden_size)
        if not isinstance(attention, nn.Module) or not isinstance(mlp, nn.Module):
            raise TypeError("attention and mlp must be constructed modules")
        if sublayer_conv is not None:
            raise ValueError("DecoderLayer does not yet support sublayer convolution")
        self.hidden_size = hidden_size
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = build_component(
            "norm", norm, hidden_size=hidden_size, device=device, dtype=dtype
        )
        self.post_attention_layernorm = build_component(
            "norm", norm, hidden_size=hidden_size, device=device, dtype=dtype
        )

    def forward(self, hidden: Tensor, inputs: LayerInputs) -> Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                "DecoderLayer requires a single [batch, sequence, hidden_size] stream"
            )
        if not isinstance(inputs, LayerInputs):
            raise TypeError("DecoderLayer requires LayerInputs")
        if inputs.attention_layout is None or inputs.attention_mask is not None:
            raise ValueError(
                "DecoderLayer requires attention_layout, not a prepared attention_mask"
            )
        # Position IDs are consumed by the backbone. Direct callers must supply
        # any required RoPE embeddings, so layers never recompute frequencies.
        update = self.self_attn(
            self.input_layernorm(hidden),
            layout=inputs.attention_layout,
            position_embeddings=inputs.position_embeddings,
            target_context=inputs.target_context,
        )
        if update.shape != hidden.shape:
            raise ValueError("attention output must preserve the residual shape")
        hidden = hidden + update
        update = self.mlp(self.post_attention_layernorm(hidden))
        if update.shape != hidden.shape:
            raise ValueError("mlp output must preserve the residual shape")
        return hidden + update
