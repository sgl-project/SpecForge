"""Compose pre-norm attention/FFN sublayers with single-stream residuals."""

from torch import Tensor, nn

from specforge.config.draft._validation import positive_int
from specforge.config.draft.norm import NormSpec

from .attention.backends import PreparedAttention
from .inputs import LayerInputs
from .registry import build_component


class DecoderLayer(nn.Module):
    """Standard residual wiring shared by model-owned and sequential execution.

    The model supplies constructed attention/FFN components. Input fusion,
    inter-layer injection, multiple residual streams, and file loading are not
    implemented here. ``sublayer_conv`` is an optional (attention, mlp) pair of
    independently constructed modules, each exposing prepare/finish. A single
    LayerSpec convolution reference supplies their shared configuration, not
    shared parameters. The layer neither builds nor clones these modules.
    """

    def __init__(
        self,
        attention: nn.Module,
        mlp: nn.Module,
        *,
        hidden_size: int,
        norm: NormSpec,
        sublayer_conv: tuple[nn.Module, nn.Module] | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        positive_int("hidden_size", hidden_size)
        if not isinstance(attention, nn.Module) or not isinstance(mlp, nn.Module):
            raise TypeError("attention and mlp must be constructed modules")
        if sublayer_conv is not None:
            if not isinstance(sublayer_conv, tuple) or len(sublayer_conv) != 2:
                raise TypeError("sublayer_conv must be an (attention, mlp) module pair")
            for module in sublayer_conv:
                if not isinstance(module, nn.Module) or not all(
                    callable(getattr(module, method, None))
                    for method in ("prepare", "finish")
                ):
                    raise TypeError(
                        "sublayer convolution modules require prepare/finish"
                    )
            if sublayer_conv[0] is sublayer_conv[1]:
                raise ValueError("attention and mlp require independent convolutions")
        self.attention_conv, self.mlp_conv = sublayer_conv or (None, None)
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
        if not isinstance(inputs.attention, PreparedAttention):
            raise ValueError("DecoderLayer requires prepared attention")
        # Position IDs are consumed by the backbone. Direct callers must supply
        # any required RoPE embeddings, so layers never recompute frequencies.
        update = self.input_layernorm(hidden)
        if self.attention_conv is not None:
            update, coefficients = self.attention_conv.prepare(update)
        update = self.self_attn(
            update,
            attention=inputs.attention,
            position_embeddings=inputs.position_embeddings,
            target_context=inputs.target_context,
        )
        if self.attention_conv is not None:
            update = self.attention_conv.finish(update, coefficients)
        if update.shape != hidden.shape:
            raise ValueError("attention output must preserve the residual shape")
        hidden = hidden + update
        update = self.post_attention_layernorm(hidden)
        if self.mlp_conv is not None:
            update, coefficients = self.mlp_conv.prepare(update)
        update = self.mlp(update)
        if self.mlp_conv is not None:
            update = self.mlp_conv.finish(update, coefficients)
        if update.shape != hidden.shape:
            raise ValueError("mlp output must preserve the residual shape")
        return hidden + update
