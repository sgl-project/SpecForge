"""Topology-free algorithm contracts and explicit registrations."""

from specforge.algorithms.contracts import (
    AlgorithmCapabilities,
    AlgorithmSpec,
    DraftRequirement,
    FeatureContract,
    FeatureMode,
    OfflineStorageContract,
)
from specforge.algorithms.registry import AlgorithmRegistration, AlgorithmRegistry

__all__ = [
    "AlgorithmCapabilities",
    "AlgorithmRegistration",
    "AlgorithmRegistry",
    "AlgorithmSpec",
    "DraftRequirement",
    "FeatureContract",
    "FeatureMode",
    "OfflineStorageContract",
]
