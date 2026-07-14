"""Topology-free contracts owned by speculative training algorithms.

These value objects describe what an algorithm consumes and supports.  They do
not describe where features come from, which process owns a model, or how a run
is launched.  Keeping those concerns out of the algorithm contract lets the
application layer resolve one deployment plan without creating another family
of topology-specific builders.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import FrozenSet, Iterable, Mapping, Protocol, Tuple

_ALGORITHM_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")


def _freeze_config_value(value: object) -> object:
    """Detach and recursively freeze common config container types."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_config_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_config_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_config_value(item) for item in value)
    return deepcopy(value)


def _normalized_names(
    values: Iterable[str],
    *,
    field_name: str,
    allow_empty: bool = False,
) -> FrozenSet[str]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be an iterable of names, not a string")
    normalized = frozenset(values)
    if not normalized and not allow_empty:
        raise ValueError(f"{field_name} must not be empty")
    invalid = [
        repr(value)
        for value in normalized
        if not isinstance(value, str) or not value or value.strip() != value
    ]
    if invalid:
        raise ValueError(
            f"{field_name} must contain non-empty names without surrounding "
            f"whitespace, got {', '.join(sorted(invalid))}"
        )
    return normalized


class FeatureMode(str, Enum):
    """How algorithm-ready target features are consumed.

    The values describe data semantics, not placement.  In particular,
    ``STREAMING`` does not imply colocated or disaggregated execution.
    """

    OFFLINE = "offline"
    STREAMING = "streaming"


class DraftConfigResolver(Protocol):
    """Validate and resolve an algorithm-specific draft configuration.

    The return type is deliberately opaque.  Algorithms may describe arbitrary
    draft structures, including mixed attention stacks, without expanding this
    shared contract every time their architecture evolves.
    """

    def __call__(self, raw_config: Mapping[str, object], /) -> object: ...


@dataclass(frozen=True)
class DraftArchitectureContract:
    """Algorithm-owned resolver for one family of draft architectures."""

    architecture: str
    resolve_config: DraftConfigResolver

    def __post_init__(self) -> None:
        if not self.architecture or self.architecture.strip() != self.architecture:
            raise ValueError(
                "architecture must be a non-empty name without surrounding whitespace"
            )
        if not callable(self.resolve_config):
            raise TypeError("resolve_config must be callable")

    def resolve(self, raw_config: Mapping[str, object]) -> object:
        """Resolve a read-only copy of an algorithm-specific config mapping."""

        if not isinstance(raw_config, Mapping):
            raise TypeError("raw_config must be a mapping")
        read_only_config = _freeze_config_value(raw_config)
        if not isinstance(read_only_config, Mapping):  # pragma: no cover - invariant
            raise TypeError("raw_config did not resolve to a mapping")
        resolved = self.resolve_config(read_only_config)
        if resolved is None:
            raise ValueError(
                f"draft config resolver for {self.architecture!r} returned None"
            )
        return resolved


@dataclass(frozen=True)
class FeatureContract:
    """Tensor-level input contract for one algorithm feature mode."""

    mode: FeatureMode
    required_tensors: FrozenSet[str]
    optional_tensors: FrozenSet[str] = frozenset()
    allowed_target_representations: FrozenSet[str] = frozenset()

    def __post_init__(self) -> None:
        try:
            mode = FeatureMode(self.mode)
        except ValueError as exc:
            raise ValueError(f"unsupported feature mode {self.mode!r}") from exc
        required = _normalized_names(
            self.required_tensors,
            field_name="required_tensors",
        )
        optional = _normalized_names(
            self.optional_tensors,
            field_name="optional_tensors",
            allow_empty=True,
        )
        representations = _normalized_names(
            self.allowed_target_representations,
            field_name="allowed_target_representations",
            allow_empty=True,
        )
        overlap = required & optional
        if overlap:
            raise ValueError(
                "required_tensors and optional_tensors must be disjoint, got "
                f"{sorted(overlap)}"
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "required_tensors", required)
        object.__setattr__(self, "optional_tensors", optional)
        object.__setattr__(self, "allowed_target_representations", representations)


@dataclass(frozen=True)
class AlgorithmCapabilities:
    """Algorithm constraints independent of deployment and feature transport."""

    modalities: FrozenSet[str]
    attention_backends: FrozenSet[str]
    required_batch_size: int | None = None
    supports_compact_teacher: bool = False
    supports_vocab_mapping: bool = False

    def __post_init__(self) -> None:
        modalities = _normalized_names(self.modalities, field_name="modalities")
        attention_backends = _normalized_names(
            self.attention_backends,
            field_name="attention_backends",
        )
        if self.required_batch_size is not None and (
            isinstance(self.required_batch_size, bool)
            or not isinstance(self.required_batch_size, int)
            or self.required_batch_size <= 0
        ):
            raise ValueError("required_batch_size must be a positive integer or None")
        for field_name in ("supports_compact_teacher", "supports_vocab_mapping"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool")
        object.__setattr__(self, "modalities", modalities)
        object.__setattr__(self, "attention_backends", attention_backends)


@dataclass(frozen=True)
class AlgorithmSpec:
    """Complete topology-free contract for one registered algorithm."""

    name: str
    draft: DraftArchitectureContract
    feature_contracts: Tuple[FeatureContract, ...]
    capabilities: AlgorithmCapabilities

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _ALGORITHM_NAME.fullmatch(self.name):
            raise ValueError(
                "algorithm name must start with a lowercase letter and contain "
                "only lowercase letters, digits, '_' or '-'"
            )
        if not isinstance(self.draft, DraftArchitectureContract):
            raise TypeError("draft must be a DraftArchitectureContract")
        if not isinstance(self.capabilities, AlgorithmCapabilities):
            raise TypeError("capabilities must be AlgorithmCapabilities")
        feature_contracts = tuple(self.feature_contracts)
        if not feature_contracts:
            raise ValueError("feature_contracts must not be empty")
        invalid = [
            type(contract).__name__
            for contract in feature_contracts
            if not isinstance(contract, FeatureContract)
        ]
        if invalid:
            raise TypeError(
                "feature_contracts must contain FeatureContract values, got "
                f"{invalid}"
            )
        modes = [contract.mode for contract in feature_contracts]
        if len(modes) != len(set(modes)):
            duplicates = sorted(
                mode.value for mode in set(modes) if modes.count(mode) > 1
            )
            raise ValueError(
                f"feature_contracts contains duplicate modes: {duplicates}"
            )
        object.__setattr__(self, "feature_contracts", feature_contracts)

    @property
    def feature_modes(self) -> FrozenSet[FeatureMode]:
        """Supported modes derived from the contracts, never duplicated."""

        return frozenset(contract.mode for contract in self.feature_contracts)

    def feature_contract(self, mode: FeatureMode | str) -> FeatureContract:
        """Return the contract for ``mode`` or raise an actionable ``KeyError``."""

        try:
            resolved_mode = FeatureMode(mode)
        except ValueError as exc:
            raise KeyError(f"unknown feature mode {mode!r}") from exc
        for contract in self.feature_contracts:
            if contract.mode is resolved_mode:
                return contract
        supported = sorted(item.value for item in self.feature_modes)
        raise KeyError(
            f"algorithm {self.name!r} does not support {resolved_mode.value!r}; "
            f"supported modes: {supported}"
        )


__all__ = [
    "AlgorithmCapabilities",
    "AlgorithmSpec",
    "DraftArchitectureContract",
    "DraftConfigResolver",
    "FeatureContract",
    "FeatureMode",
]
