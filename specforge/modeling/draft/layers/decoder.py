"""Compose pre-norm sublayers through independently owned residual connections."""

from torch import Tensor, nn

from specforge.config.draft._validation import positive_int
from specforge.config.draft.norm import NormSpec

from .attention.backends import PreparedAttention
from .inputs import ExpertChoice, LayerInputs, MoELayerOutput, MoEOutput
from .registry import build_component
from .residual import ResidualConnection


class DecoderLayer(nn.Module):
    """The caller supplies attention, FFN and one residual connection per sublayer.

    Connections own stream layout and mixing. Their callable includes normalization
    and optional convolution, and consumes one [batch, sequence, hidden_size] stream.
    ``sublayer_conv`` contains independently constructed attention/FFN modules with
    prepare/finish methods; the decoder neither builds nor clones them.
    """

    def __init__(
        self,
        attention: nn.Module,
        mlp: nn.Module,
        *,
        hidden_size: int,
        norm: NormSpec,
        attention_connection: ResidualConnection,
        mlp_connection: ResidualConnection,
        sublayer_conv: tuple[nn.Module, nn.Module] | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        positive_int("hidden_size", hidden_size)
        if not isinstance(attention, nn.Module) or not isinstance(mlp, nn.Module):
            raise TypeError("attention and mlp must be constructed modules")
        if not all(
            isinstance(connection, ResidualConnection)
            for connection in (attention_connection, mlp_connection)
        ):
            raise TypeError("sublayer connections must be ResidualConnection modules")
        if attention_connection is mlp_connection:
            raise ValueError(
                "attention and mlp require independent residual connections"
            )
        self.attention_connection = attention_connection
        self.mlp_connection = mlp_connection
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

    def forward(self, hidden: Tensor, inputs: LayerInputs) -> Tensor | MoELayerOutput:
        """The stream, or with a MoE FFN the typed layer output with its expert choice."""
        if not isinstance(inputs, LayerInputs):
            raise TypeError("DecoderLayer requires LayerInputs")
        if not isinstance(inputs.attention, PreparedAttention):
            raise ValueError("DecoderLayer requires prepared attention")
        hidden = self.attention_connection(
            hidden, lambda stream: self._attention_update(stream, inputs)
        )
        choices: list[ExpertChoice] = []

        def mlp_update(stream: Tensor) -> Tensor:
            update, choice = self._mlp_update(stream)
            if choice is not None:
                choices.append(choice)
            return update

        hidden = self.mlp_connection(hidden, mlp_update)
        if not choices:
            return hidden
        if len(choices) > 1:
            raise ValueError("a MoE FFN needs a residual connection that calls it once")
        return MoELayerOutput(hidden=hidden, expert_choice=choices[0])

    def _check_stream(self, hidden: Tensor) -> None:
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                "Decoder sublayers require a single [batch, sequence, hidden_size] stream"
            )

    @staticmethod
    def _check_update(update: Tensor, stream: Tensor, sublayer: str) -> Tensor:
        # a multi-stream connection would broadcast a mis-shaped update silently
        if not isinstance(update, Tensor) or update.shape != stream.shape:
            raise ValueError(f"{sublayer} output must preserve the stream shape")
        return update

    def _attention_update(self, hidden: Tensor, inputs: LayerInputs) -> Tensor:
        self._check_stream(hidden)
        # Position IDs are consumed by the backbone, not individual layers.
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
        return self._check_update(update, hidden, "attention")

    def _mlp_update(self, hidden: Tensor) -> tuple[Tensor, ExpertChoice | None]:
        self._check_stream(hidden)
        update = self.post_attention_layernorm(hidden)
        if self.mlp_conv is not None:
            update, coefficients = self.mlp_conv.prepare(update)
        result = self.mlp(update)
        choice = result.expert_choice if isinstance(result, MoEOutput) else None
        update = result.update if choice is not None else result
        if self.mlp_conv is not None:
            update = self.mlp_conv.finish(update, coefficients)
        return self._check_update(update, hidden, "mlp"), choice
