"""The immutable effective draft structure; HF and artifact metadata live outside."""

from collections.abc import Mapping
from dataclasses import dataclass

from ._validation import boolean, finite_number, positive_int
from .architecture import ArchitectureSpec
from .attention import AttentionSpec
from .conv import ConvSpec
from .immutable import FrozenMap
from .layer import LayerSpec
from .logits import TargetLogitSpec
from .mlp import FFNSpec
from .norm import NormSpec
from .residual import ResidualSpec
from .rope import RopeSpec
from .validate import validate


@dataclass(frozen=True, kw_only=True)
class DraftConfig:
    architecture: str
    hidden_size: int
    vocab_size: int
    draft_vocab_size: int
    max_position_embeddings: int
    initializer_range: float
    tie_word_embeddings: bool
    attention_configs: FrozenMap[AttentionSpec]
    mlp_configs: FrozenMap[FFNSpec]
    conv_configs: FrozenMap[ConvSpec]
    layer_configs: FrozenMap[LayerSpec]
    layer_types: tuple[str, ...]
    norm_config: NormSpec
    residual_config: ResidualSpec
    rope_parameters: RopeSpec
    spec_config: ArchitectureSpec
    target_bindings: tuple[str, ...]
    target_logit_transform: TargetLogitSpec | None

    def __post_init__(self):
        for name in (
            "hidden_size",
            "vocab_size",
            "draft_vocab_size",
            "max_position_embeddings",
        ):
            positive_int(name, getattr(self, name))
        finite_number("initializer_range", self.initializer_range)
        if self.initializer_range < 0:
            raise ValueError("initializer_range must be nonnegative")
        boolean("tie_word_embeddings", self.tie_word_embeddings)
        for name in (
            "attention_configs",
            "mlp_configs",
            "conv_configs",
            "layer_configs",
        ):
            values = getattr(self, name)
            if not isinstance(values, Mapping):
                raise ValueError(f"{name} must be a mapping")
            object.__setattr__(self, name, FrozenMap(values))
        for name, schema in (
            ("norm_config", NormSpec),
            ("residual_config", ResidualSpec),
            ("rope_parameters", RopeSpec),
            ("spec_config", ArchitectureSpec),
        ):
            if not isinstance(getattr(self, name), schema):
                raise ValueError(f"{name} must be a {schema.__name__}")
        if type(self.target_bindings) is not tuple or any(
            type(name) is not str or name not in {"embed_tokens", "lm_head"}
            for name in self.target_bindings
        ):
            raise ValueError(
                "target_bindings must be a tuple containing only embed_tokens and lm_head"
            )
        if len(set(self.target_bindings)) != len(self.target_bindings):
            raise ValueError("target_bindings must not contain duplicates")
        object.__setattr__(self, "target_bindings", tuple(sorted(self.target_bindings)))
        validate(self)
        try:
            hash(self)
        except TypeError as exc:
            raise ValueError(
                "draft configuration nodes must be immutable and hashable"
            ) from exc

    @property
    def num_hidden_layers(self) -> int:
        return len(self.layer_types)
