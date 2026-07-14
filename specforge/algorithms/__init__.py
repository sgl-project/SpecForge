"""Algorithm-owned contracts for speculative draft training.

The package intentionally contains no runtime, deployment, or model-loading
policy.  Concrete algorithms register their contracts with the application
composition root; transport and process topology are resolved elsewhere.
"""

from specforge.algorithms.contracts import (
    AlgorithmCapabilities,
    AlgorithmSpec,
    DraftArchitectureContract,
    DraftConfigResolver,
    FeatureContract,
    FeatureMode,
)
from specforge.algorithms.registry import AlgorithmRegistry

__all__ = [
    "AlgorithmCapabilities",
    "AlgorithmRegistry",
    "AlgorithmSpec",
    "DraftArchitectureContract",
    "DraftConfigResolver",
    "FeatureContract",
    "FeatureMode",
]
