# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Core dataflow contracts shared between SpecForge components.

These records are intentionally small. They describe *what* components exchange,
not how any backend is implemented. The module is deliberately dependency-light:
it imports only the standard library so the control plane can be reasoned about
(and unit-tested) without pulling in torch or the heavy model code.

The single load-bearing invariant: control-plane records (``PromptTask``,
``SampleRef``) carry **metadata only** — never tensors. Large tensors move
through the data plane (``FeatureStore``) and surface only inside ``TrainBatch``
on the trainer side. ``assert_no_tensors`` makes that invariant checkable.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a hard torch dependency
    import torch

# Schema version bumped whenever the on-the-wire feature schema changes in a way
# that makes older SampleRef records unreadable. Loaders gate on this.
SCHEMA_VERSION = 1

RunMode = Literal["online", "offline"]
DeploymentMode = Literal["local_colocated", "dataflow_colocated", "disaggregated"]
DraftStrategyName = Literal["eagle3", "dflash"]
# Tagged union for the EAGLE3 target feature. The *strategy* owns the
# projection so the trainer core stays branch-free:
#   - pruned_logits: rollout applied the t2d vocab map; stored (seq, draft_vocab)
#   - logits:        full (seq, target_vocab); parity/debug only
#   - hidden_state:  target last hidden state; strategy re-runs lm_head + t2d
TargetRepr = Literal["logits", "pruned_logits", "hidden_state"]


@dataclass(frozen=True)
class PromptTask:
    """A unit of work handed to a rollout worker. Metadata only."""

    task_id: str
    run_id: str
    source_id: str
    payload: Dict[str, Any]  # conversation, preformatted text, or token IDs
    max_length: int
    chat_template: Optional[str] = None
    loss_mask_policy: Dict[str, Any] = field(default_factory=dict)
    target_model_version: str = "unknown"
    draft_weight_version: Optional[str] = None
    attempt: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureSpec:
    """Describes one named tensor that lives in the feature store.

    Shape/dtype are metadata; the tensor itself never travels with the spec.
    """

    name: str  # input_ids, hidden_states, target, loss_mask, ...
    shape: Tuple[int, ...]
    dtype: str
    device_hint: Optional[str] = None
    required: bool = True
    target_repr: Optional[TargetRepr] = None
    # vocab map / head version / softmax convention — only meaningful for the
    # `target` feature, and mandatory when target_repr == "hidden_state".
    target_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SampleRef:
    """A pointer to one training sample's features. Metadata only — no tensors.

    Exactly one sample per ref (batching is a loader concern, never baked in).
    """

    sample_id: str
    run_id: str
    source_task_id: Optional[str]
    feature_store_uri: str
    feature_keys: Dict[str, str]
    feature_specs: Dict[str, FeatureSpec]
    strategy: DraftStrategyName
    schema_version: int = SCHEMA_VERSION
    target_model_version: str = "unknown"
    draft_weight_version: Optional[str] = None
    tokenizer_version: str = "unknown"
    num_tokens: int = 0
    estimated_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureHandle:
    """Lifetime token returned by ``FeatureStore.get``.

    ``generation`` is bumped on every (re)materialization of a sample so a stale
    ``release`` is a safe no-op. ``lease_token`` is opaque and required to
    release. The local in-memory backend uses a trivial handle, carrying the
    contract without paying for it.
    """

    sample_id: str
    generation: int
    lease_token: str


@dataclass
class TrainBatch:
    """A materialized, collated batch ready for the trainer. Holds tensors.

    This is the *only* contract that carries tensors, and only ever on the
    trainer / data-plane side.
    """

    sample_ids: List[str]
    strategy: DraftStrategyName
    tensors: Dict[str, "torch.Tensor"]
    metadata: Dict[str, Any] = field(default_factory=dict)


# NOTE: the published-weight lifecycle (WeightVersion, WeightPublisher, hot
# update, serving accept-length gate) is not implemented here — it is not needed
# for the local train pipeline. SampleRef/PromptTask still carry a
# ``draft_weight_version`` *string* as rollout provenance, but there is no
# WeightVersion object or publisher here yet.


# ---------------------------------------------------------------------------
# No-tensor invariant
# ---------------------------------------------------------------------------
def _looks_like_tensor(obj: Any) -> bool:
    """Duck-typed tensor / ndarray detection without importing torch/numpy."""
    cls = type(obj)
    module = getattr(cls, "__module__", "") or ""
    root = module.split(".", 1)[0]
    if root in ("torch", "numpy"):
        return True
    # torch.Tensor and np.ndarray both expose these; plain containers do not.
    return hasattr(obj, "dtype") and hasattr(obj, "shape") and hasattr(obj, "device")


def assert_no_tensors(obj: Any, *, _path: str = "<root>") -> None:
    """Recursively assert that ``obj`` carries no tensor payloads.

    Used by the control plane to enforce that ``PromptTask`` / ``SampleRef``
    records (including their ``metadata``) never smuggle a tensor through a
    controller API. ``test_controller_carries_no_tensor`` exercises this.
    """
    if obj is None or isinstance(obj, (str, bytes, bool, int, float)):
        return
    if _looks_like_tensor(obj):
        raise TypeError(
            f"tensor payload found at {_path}: control-plane records must carry "
            f"metadata only (type={type(obj).__module__}.{type(obj).__name__})"
        )
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            assert_no_tensors(getattr(obj, f.name), _path=f"{_path}.{f.name}")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert_no_tensors(v, _path=f"{_path}[{k!r}]")
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for i, v in enumerate(obj):
            assert_no_tensors(v, _path=f"{_path}[{i}]")
        return
    # Other scalar/opaque metadata (e.g. a version string container) is fine.
    return


__all__ = [
    "SCHEMA_VERSION",
    "RunMode",
    "DeploymentMode",
    "DraftStrategyName",
    "TargetRepr",
    "PromptTask",
    "FeatureSpec",
    "SampleRef",
    "FeatureHandle",
    "TrainBatch",
    "assert_no_tensors",
]
