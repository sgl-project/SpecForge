"""Feed-forward structure, serialized as mlp_configs."""

from dataclasses import dataclass
from typing import ClassVar

from .._validation import nonempty_string, positive_int
from .base import FFNSpec


@dataclass(frozen=True, kw_only=True)
class DenseFFNSpec(FFNSpec):
    type: ClassVar[str] = "dense"

    intermediate_size: int
    hidden_act: str

    def __post_init__(self):
        positive_int("intermediate_size", self.intermediate_size)
        nonempty_string("hidden_act", self.hidden_act)
