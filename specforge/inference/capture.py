# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""CaptureConfig: the typed contract for what a rollout must extract (B7/B8).

``capture`` is NOT an untyped ``dict[str, Any]``. It is a frozen config *derived
from the active strategy* (``feature_names == DraftTrainStrategy.required_features``,
``aux_hidden_state_layer_ids`` == the layers the draft config requested). Before
any ``FeatureStore.put`` the rollout runs :func:`verify_capture`, which fails
loudly on a name / aux-layer-id / width / target-dim mismatch — turning what
would otherwise be a confusing downstream trainer bug into an immediate,
localized error at the extraction boundary.

Import-light (stdlib only) so the assertions are unit-testable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional, Tuple

from specforge.runtime.contracts import TargetRepr


class CaptureMismatchError(AssertionError):
    """Raised when extracted features do not match the requested CaptureConfig."""


@dataclass(frozen=True)
class CaptureConfig:
    feature_names: FrozenSet[str]
    aux_hidden_state_layer_ids: Tuple[int, ...]
    target_repr: TargetRepr
    target_hidden_size: int
    target_vocab_size: Optional[int] = None
    draft_vocab_size: Optional[int] = None
    vocab_map_version: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_strategy(
        cls,
        required_features,
        aux_hidden_state_layer_ids,
        *,
        target_repr: TargetRepr,
        target_hidden_size: int,
        target_vocab_size: Optional[int] = None,
        draft_vocab_size: Optional[int] = None,
        vocab_map_version: Optional[str] = None,
    ) -> "CaptureConfig":
        return cls(
            feature_names=frozenset(required_features),
            aux_hidden_state_layer_ids=tuple(aux_hidden_state_layer_ids),
            target_repr=target_repr,
            target_hidden_size=int(target_hidden_size),
            target_vocab_size=target_vocab_size,
            draft_vocab_size=draft_vocab_size,
            vocab_map_version=vocab_map_version,
        )

    @property
    def expected_aux_width(self) -> int:
        return len(self.aux_hidden_state_layer_ids) * self.target_hidden_size

    def expected_target_dim(self) -> Optional[int]:
        if self.target_repr == "pruned_logits":
            return self.draft_vocab_size
        if self.target_repr == "logits":
            return self.target_vocab_size
        if self.target_repr == "hidden_state":
            return self.target_hidden_size
        return None


def verify_capture(
    tensors: Dict[str, Any],
    capture: CaptureConfig,
    *,
    sample_id: str,
    recorded_aux_layer_ids: Optional[Tuple[int, ...]] = None,
    aux_feature_name: str = "hidden_state",
    target_feature_name: str = "target",
) -> None:
    """Loud, pre-put validation that extracted ``tensors`` match ``capture``.

    Raises :class:`CaptureMismatchError` on the first mismatch with a
    requested-vs-actual diff and the offending ``sample_id``.
    """
    # (1) all requested feature names present
    missing = sorted(n for n in capture.feature_names if n not in tensors)
    if missing:
        raise CaptureMismatchError(
            f"[{sample_id}] capture missing features {missing}; "
            f"got {sorted(tensors)}; requested {sorted(capture.feature_names)}"
        )

    # (2) recorded aux-layer IDs == requested
    if recorded_aux_layer_ids is not None:
        if tuple(recorded_aux_layer_ids) != capture.aux_hidden_state_layer_ids:
            raise CaptureMismatchError(
                f"[{sample_id}] aux-layer id mismatch: recorded "
                f"{tuple(recorded_aux_layer_ids)} != requested "
                f"{capture.aux_hidden_state_layer_ids}"
            )

    # (3) aux width == len(aux_layer_ids) * target_hidden_size
    if aux_feature_name in tensors and capture.aux_hidden_state_layer_ids:
        width = int(tuple(tensors[aux_feature_name].shape)[-1])
        if width != capture.expected_aux_width:
            raise CaptureMismatchError(
                f"[{sample_id}] aux width {width} != "
                f"len(aux_layer_ids)*target_hidden_size="
                f"{len(capture.aux_hidden_state_layer_ids)}*"
                f"{capture.target_hidden_size}={capture.expected_aux_width}"
            )

    # (4) target last-dim matches target_repr (+ vocab-map dim for pruned_logits)
    expected = capture.expected_target_dim()
    if target_feature_name in tensors and expected is not None:
        dim = int(tuple(tensors[target_feature_name].shape)[-1])
        if dim != expected:
            raise CaptureMismatchError(
                f"[{sample_id}] target last-dim {dim} != expected {expected} "
                f"for target_repr={capture.target_repr!r}"
            )
        if capture.target_repr == "pruned_logits" and capture.vocab_map_version is None:
            raise CaptureMismatchError(
                f"[{sample_id}] target_repr='pruned_logits' requires a "
                f"vocab_map_version so the trainer-side mapping is gated"
            )


__all__ = ["CaptureConfig", "CaptureMismatchError", "verify_capture"]
