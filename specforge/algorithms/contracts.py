"""Pure contracts describing what a speculative training algorithm consumes.

This module intentionally contains no factories, model classes, or runtime
objects.  Algorithm implementations live behind provider ports; deployment and
transport are resolved by the application layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import FrozenSet, Iterable, Tuple

_ALGORITHM_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")


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


def _assert_pure_value(value: object, *, path: str) -> None:
    """Reject executable or opaque state recursively from public contracts."""

    if isinstance(value, type) or callable(value):
        raise TypeError(f"{path} must be a pure value, got executable {value!r}")
    if value is None or isinstance(value, (str, int, float, bool, Enum)):
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        for index, item in enumerate(value):
            _assert_pure_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_pure_value(key, path=f"{path}.key")
            _assert_pure_value(item, path=f"{path}[{key!r}]")
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _assert_pure_value(
                getattr(value, field.name),
                path=f"{path}.{field.name}",
            )
        return
    raise TypeError(
        f"{path} must contain only serializable contract values, got "
        f"{type(value).__name__}"
    )


class FeatureMode(str, Enum):
    """How algorithm-ready target features are consumed."""

    OFFLINE = "offline"
    STREAMING = "streaming"


@dataclass(frozen=True)
class DraftRequirement:
    """Pure compatibility requirements for a draft architecture.

    Draft configuration loading, target-derived generation, validation, and
    model construction belong to the draft-model registry and algorithm-owned
    providers.  Only stable identifiers and declarative override names belong
    here.
    """

    compatible_architectures: FrozenSet[str]
    default_architecture: str
    supported_overrides: FrozenSet[str] = frozenset()
    fixed_override_values: Tuple[Tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        architectures = _normalized_names(
            self.compatible_architectures,
            field_name="compatible_architectures",
        )
        overrides = _normalized_names(
            self.supported_overrides,
            field_name="supported_overrides",
            allow_empty=True,
        )
        if self.default_architecture not in architectures:
            raise ValueError(
                "default_architecture must be present in " "compatible_architectures"
            )
        fixed_values = tuple(self.fixed_override_values)
        fixed_names = [name for name, _value in fixed_values]
        duplicates = sorted(
            name for name in set(fixed_names) if fixed_names.count(name) > 1
        )
        if duplicates:
            raise ValueError(f"duplicate fixed override values: {duplicates}")
        unsupported = sorted(set(fixed_names) - overrides)
        if unsupported:
            raise ValueError(
                "fixed_override_values must name supported overrides, got "
                f"{unsupported}"
            )
        if any(
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            for name, value in fixed_values
        ):
            raise TypeError("fixed_override_values must contain (name, integer) pairs")
        object.__setattr__(self, "compatible_architectures", architectures)
        object.__setattr__(self, "supported_overrides", overrides)
        object.__setattr__(self, "fixed_override_values", fixed_values)


@dataclass(frozen=True)
class OfflineStorageContract:
    """Raw offline record shape before an algorithm normalizer is applied."""

    format: str
    required_tensors: FrozenSet[str]
    normalizer: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        for field_name in ("format", "normalizer"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(
                    f"{field_name} must be a non-empty name without surrounding "
                    "whitespace"
                )
        if self.schema_version != 1:
            raise ValueError("only offline storage schema_version=1 is supported")
        object.__setattr__(
            self,
            "required_tensors",
            _normalized_names(
                self.required_tensors,
                field_name="required_tensors",
            ),
        )


@dataclass(frozen=True)
class FeatureContract:
    """Algorithm-ready tensors for one ``(mode, modality)`` pair."""

    mode: FeatureMode
    modality: str
    required_tensors: FrozenSet[str]
    optional_tensors: FrozenSet[str] = frozenset()
    allowed_target_representations: FrozenSet[str] = frozenset()
    default_target_representation: str | None = None
    schema_version: int = 1
    storage: OfflineStorageContract | None = None

    def __post_init__(self) -> None:
        try:
            mode = FeatureMode(self.mode)
        except ValueError as exc:
            raise ValueError(f"unsupported feature mode {self.mode!r}") from exc
        if (
            not isinstance(self.modality, str)
            or not self.modality
            or self.modality.strip() != self.modality
        ):
            raise ValueError(
                "modality must be a non-empty name without surrounding whitespace"
            )
        if self.schema_version != 1:
            raise ValueError("only feature schema_version=1 is supported")
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
        if representations:
            if self.default_target_representation not in representations:
                raise ValueError(
                    "default_target_representation must be present in "
                    "allowed_target_representations"
                )
        elif self.default_target_representation is not None:
            raise ValueError(
                "default_target_representation requires at least one allowed "
                "target representation"
            )
        if mode is FeatureMode.OFFLINE and self.storage is None:
            raise ValueError("offline feature contracts require a storage contract")
        if mode is FeatureMode.STREAMING and self.storage is not None:
            raise ValueError("streaming feature contracts cannot define storage")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "required_tensors", required)
        object.__setattr__(self, "optional_tensors", optional)
        object.__setattr__(self, "allowed_target_representations", representations)

    @property
    def key(self) -> tuple[FeatureMode, str]:
        return self.mode, self.modality


@dataclass(frozen=True)
class AlgorithmCapabilities:
    """Static algorithm constraints independent of deployment topology."""

    attention_backends: FrozenSet[str]
    required_batch_size: int | None = None
    supports_compact_teacher: bool = False
    supports_vocab_mapping: bool = False
    allows_aux_layer_override: bool = False

    def __post_init__(self) -> None:
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
        for field_name in (
            "supports_compact_teacher",
            "supports_vocab_mapping",
            "allows_aux_layer_override",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool")
        object.__setattr__(self, "attention_backends", attention_backends)


@dataclass(frozen=True)
class AlgorithmSpec:
    """Complete topology-free, executable-free contract for one algorithm."""

    name: str
    draft: DraftRequirement
    feature_contracts: Tuple[FeatureContract, ...]
    capabilities: AlgorithmCapabilities

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _ALGORITHM_NAME.fullmatch(self.name):
            raise ValueError(
                "algorithm name must start with a lowercase letter and contain "
                "only lowercase letters, digits, '_' or '-'"
            )
        if not isinstance(self.draft, DraftRequirement):
            raise TypeError("draft must be a DraftRequirement")
        if not isinstance(self.capabilities, AlgorithmCapabilities):
            raise TypeError("capabilities must be AlgorithmCapabilities")
        contracts = tuple(self.feature_contracts)
        if not contracts:
            raise ValueError("feature_contracts must not be empty")
        invalid = [
            type(contract).__name__
            for contract in contracts
            if not isinstance(contract, FeatureContract)
        ]
        if invalid:
            raise TypeError(
                "feature_contracts must contain FeatureContract values, got "
                f"{invalid}"
            )
        keys = [contract.key for contract in contracts]
        duplicates = sorted(
            (mode.value, modality)
            for mode, modality in set(keys)
            if keys.count((mode, modality)) > 1
        )
        if duplicates:
            raise ValueError(
                "feature_contracts contains duplicate (mode, modality) keys: "
                f"{duplicates}"
            )
        object.__setattr__(self, "feature_contracts", contracts)
        _assert_pure_value(self, path="AlgorithmSpec")

    @property
    def modalities(self) -> FrozenSet[str]:
        return frozenset(contract.modality for contract in self.feature_contracts)

    @property
    def feature_modes(self) -> FrozenSet[FeatureMode]:
        return frozenset(contract.mode for contract in self.feature_contracts)

    @property
    def supports_online(self) -> bool:
        """Whether any streaming feature contract is declared."""

        return FeatureMode.STREAMING in self.feature_modes

    def supports(self, mode: FeatureMode | str, modality: str) -> bool:
        try:
            resolved_mode = FeatureMode(mode)
        except ValueError:
            return False
        return any(
            contract.key == (resolved_mode, modality)
            for contract in self.feature_contracts
        )

    def feature_contract(
        self,
        mode: FeatureMode | str,
        modality: str,
    ) -> FeatureContract:
        try:
            resolved_mode = FeatureMode(mode)
        except ValueError as exc:
            raise KeyError(f"unknown feature mode {mode!r}") from exc
        for contract in self.feature_contracts:
            if contract.key == (resolved_mode, modality):
                return contract
        supported = sorted(
            (contract.mode.value, contract.modality)
            for contract in self.feature_contracts
        )
        raise KeyError(
            f"algorithm {self.name!r} does not support "
            f"({resolved_mode.value!r}, {modality!r}); supported: {supported}"
        )


__all__ = [
    "AlgorithmCapabilities",
    "AlgorithmSpec",
    "DraftRequirement",
    "FeatureContract",
    "FeatureMode",
    "OfflineStorageContract",
]
