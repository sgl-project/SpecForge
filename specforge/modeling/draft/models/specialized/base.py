"""Native checkpoint adapter contract; independent of model forward methods."""

from collections.abc import Mapping
from typing import ClassVar, Protocol

from torch import Tensor


class NativeDraftAdapter(Protocol):
    native_prefix: ClassVar[str]
    target_embedding_keys: ClassVar[tuple[str, ...]]
    target_head_keys: ClassVar[tuple[str, ...]]

    @classmethod
    def native_state_dict(cls, state: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Map a trained draft state to the native serving key layout."""
        ...

    @classmethod
    def required_keys(
        cls, state: Mapping[str, Tensor], *, share_lm_head: bool
    ) -> set[str]:
        """Required native tensors after accounting for externally shared weights."""
        ...

    @classmethod
    def allowed_extra_keys(cls, *, share_lm_head: bool) -> set[str]:
        """Native tensors allowed but not owned by the trainable draft."""
        ...

    @classmethod
    def patch_target_config(cls, base_config: dict, draft_config: dict) -> dict:
        """Adapt native target config fields needed to serve the trained draft."""
        ...
