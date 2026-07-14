# coding=utf-8
"""SpecForge DataFlow runtime.

A DataFlow-centered layer over the existing SpecForge model/data code:

    PromptTask -> RolloutWorker -> SampleRef -> FeatureDataLoader -> TrainBatch -> Trainer

The control plane (controller, queues) moves only metadata; large tensors move
only through the data plane (FeatureStore). This module re-exports the
dependency-light contracts. The compute planes live in the top-level
``specforge.training`` and ``specforge.inference`` packages and are imported
explicitly by callers, not at runtime package load.
"""

from specforge.runtime.contracts import (  # noqa: F401
    SCHEMA_VERSION,
    FeatureHandle,
    FeatureSpec,
    PromptTask,
    SampleRef,
    TrainBatch,
    assert_no_tensors,
)

__all__ = [
    "SCHEMA_VERSION",
    "PromptTask",
    "FeatureSpec",
    "SampleRef",
    "FeatureHandle",
    "TrainBatch",
    "assert_no_tensors",
]
