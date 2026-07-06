# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""FeatureContract: the typed contract for rollout-produced feature records.

This module is the runtime/data-plane counterpart to
``target_engine.capture_policy``. Target-capture policies define how a backend
produces a typed batched target output. A ``FeatureContract`` defines what the
runtime adapter must turn that output into before any ``FeatureStore.put``:
feature names, aux-layer ids, target representation, and expected dimensions.

Before any store write the rollout runs :func:`verify_feature_contract`, which
fails loudly on a name / aux-layer-id / width / target-dim mismatch — turning
what would otherwise be a confusing downstream trainer bug into an immediate,
localized error at the extraction boundary.

Import-light (stdlib only) so the assertions are unit-testable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional, Tuple

from specforge.runtime.contracts import TargetRepr


class FeatureContractError(AssertionError):
    """Raised when extracted features do not match the requested contract."""


@dataclass(frozen=True)
class FeatureContract:
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
    ) -> "FeatureContract":
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


def verify_feature_contract(
    tensors: Dict[str, Any],
    contract: FeatureContract,
    *,
    sample_id: str,
    recorded_aux_layer_ids: Optional[Tuple[int, ...]] = None,
    aux_feature_name: str = "hidden_state",
    target_feature_name: str = "target",
) -> None:
    """Loud, pre-put validation that extracted ``tensors`` match ``contract``.

    Raises :class:`FeatureContractError` on the first mismatch with a
    requested-vs-actual diff and the offending ``sample_id``.
    """
    # (1) all requested feature names present
    missing = sorted(n for n in contract.feature_names if n not in tensors)
    if missing:
        raise FeatureContractError(
            f"[{sample_id}] capture missing features {missing}; "
            f"got {sorted(tensors)}; requested {sorted(contract.feature_names)}"
        )

    # (2) recorded aux-layer IDs == requested
    if recorded_aux_layer_ids is not None:
        if tuple(recorded_aux_layer_ids) != contract.aux_hidden_state_layer_ids:
            raise FeatureContractError(
                f"[{sample_id}] aux-layer id mismatch: recorded "
                f"{tuple(recorded_aux_layer_ids)} != requested "
                f"{contract.aux_hidden_state_layer_ids}"
            )

    # (3) aux width == len(aux_layer_ids) * target_hidden_size
    if aux_feature_name in tensors and contract.aux_hidden_state_layer_ids:
        width = int(tuple(tensors[aux_feature_name].shape)[-1])
        if width != contract.expected_aux_width:
            raise FeatureContractError(
                f"[{sample_id}] aux width {width} != "
                f"len(aux_layer_ids)*target_hidden_size="
                f"{len(contract.aux_hidden_state_layer_ids)}*"
                f"{contract.target_hidden_size}={contract.expected_aux_width}"
            )

    # (4) target last-dim matches target_repr (+ vocab-map dim for pruned_logits)
    expected = contract.expected_target_dim()
    if target_feature_name in tensors and expected is not None:
        dim = int(tuple(tensors[target_feature_name].shape)[-1])
        if dim != expected:
            raise FeatureContractError(
                f"[{sample_id}] target last-dim {dim} != expected {expected} "
                f"for target_repr={contract.target_repr!r}"
            )
        if (
            contract.target_repr == "pruned_logits"
            and contract.vocab_map_version is None
        ):
            raise FeatureContractError(
                f"[{sample_id}] target_repr='pruned_logits' requires a "
                f"vocab_map_version so the trainer-side mapping is gated"
            )


def verify_feature_contract_specs(
    specs: Dict[str, Any],
    contract: FeatureContract,
    *,
    sample_id: str,
    recorded_aux_layer_ids: Optional[Tuple[int, ...]] = None,
    aux_feature_name: str = "hidden_state",
    target_feature_name: str = "target",
) -> None:
    """Contract verification from ``FeatureSpec``s alone — no tensors.

    Used by ref-producing sources (server-side capture): the tensors already
    live in the store, so the extraction-boundary checks run against the
    returned shape/dtype metadata. Every check in
    :func:`verify_feature_contract` reads only ``.shape`` from the mapping's
    values, which ``FeatureSpec`` provides — so this is the same validation,
    same error messages, same loudness.
    """
    verify_feature_contract(
        specs,
        contract,
        sample_id=sample_id,
        recorded_aux_layer_ids=recorded_aux_layer_ids,
        aux_feature_name=aux_feature_name,
        target_feature_name=target_feature_name,
    )


def verify_capture(
    tensors: Dict[str, Any],
    capture: FeatureContract,
    *,
    sample_id: str,
    recorded_aux_layer_ids: Optional[Tuple[int, ...]] = None,
    aux_feature_name: str = "hidden_state",
    target_feature_name: str = "target",
) -> None:
    """Back-compat alias for :func:`verify_feature_contract`."""
    verify_feature_contract(
        tensors,
        capture,
        sample_id=sample_id,
        recorded_aux_layer_ids=recorded_aux_layer_ids,
        aux_feature_name=aux_feature_name,
        target_feature_name=target_feature_name,
    )


# Back-compat aliases. New runtime code should prefer FeatureContract names to
# distinguish feature-store contracts from target-capture policies.
CaptureConfig = FeatureContract
CaptureMismatchError = FeatureContractError


__all__ = [
    "FeatureContract",
    "FeatureContractError",
    "verify_feature_contract",
    "verify_feature_contract_specs",
    "CaptureConfig",
    "CaptureMismatchError",
    "verify_capture",
]
