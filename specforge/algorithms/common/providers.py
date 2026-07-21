"""Executable provider ports for built-in algorithms.

The public :class:`~specforge.algorithms.contracts.AlgorithmSpec` deliberately
contains values only.  This module is the executable half of one
``AlgorithmRegistration``: step/model factories and data adapters selected by
the application composition root.

The provider types describe algorithm-owned behavior, never deployment
topology.  In particular, ``ServerStreamingProvider`` means that an algorithm
can consume features captured by an external server; it does not select a
backend, start a server, or construct a transport.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    FrozenSet,
    Iterable,
    Mapping,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from specforge.algorithms.contracts import AlgorithmSpec, FeatureMode
from specforge.algorithms.registry import AlgorithmRegistration

Factory = Callable[..., Any]

STEP_OPTIONS_CONTRACT_KEY = "specforge_step_options"
MODEL_PROVENANCE_CONTRACT_KEY = "specforge_model_provenance"
OMITTED_STATE_FINGERPRINT_CONTRACT_KEY = (
    "specforge_omitted_checkpoint_state_fingerprint"
)


@runtime_checkable
class ServerInputAdapter(Protocol):
    """Modality-owned input port for external server capture.

    Implementations may load a tokenizer, processor, or any other immutable
    input tooling; turn configured source data into JSON-safe prompt payloads;
    and map a batch of those payloads onto the SGLang ``/generate`` request.
    The transport owns capture metadata and sampling parameters, so adapters
    return only model-input fields.
    """

    def load_input_tools(self, config: Any) -> Any:
        """Load tokenizer/processor tooling required by this modality."""

    def prepare_prompts(
        self,
        config: Any,
        input_tools: Any,
        *,
        draft_config: Any,
    ) -> list[dict[str, Any]]:
        """Build JSON-safe prompt dictionaries for the producer runtime."""

    def build_request_inputs(
        self,
        tasks: Sequence[Any],
    ) -> Mapping[str, Any]:
        """Build model-input fields for one batched server request."""


def _non_empty(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(
            f"{field_name} must be a non-empty name without surrounding whitespace"
        )
    return value


def _factories_are_callable(values: Iterable[tuple[str, object]]) -> None:
    for field_name, value in values:
        if not callable(value):
            raise TypeError(f"{field_name} must be callable")


def checkpoint_key_fingerprint(model: Any, keys: Iterable[str]) -> str:
    """Hash selected state tensors with bounded host memory.

    Providers may intentionally omit reconstructable frozen tensors from a
    checkpoint. Their values still affect the model, so resume records this
    fingerprint and rejects a reconstruction from different weights.
    """

    import torch

    state = model.state_dict()
    hasher = hashlib.sha256()
    for key in sorted(keys):
        if key not in state:
            raise ValueError(
                f"cannot fingerprint missing checkpoint-policy key {key!r}"
            )
        tensor = state[key]
        if not torch.is_tensor(tensor):
            raise TypeError(
                f"checkpoint-policy key {key!r} is not a tensor: "
                f"{type(tensor).__name__}"
            )
        hasher.update(key.encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("ascii"))
        hasher.update(repr(tuple(tensor.shape)).encode("ascii"))
        flat = tensor.detach().contiguous().view(-1)
        chunk_elements = max(1, (16 * 1024 * 1024) // flat.element_size())
        for start in range(0, flat.numel(), chunk_elements):
            chunk = flat[start : start + chunk_elements].to(
                device="cpu",
                copy=True,
            )
            hasher.update(memoryview(chunk.view(torch.uint8).numpy()))
    return hasher.hexdigest()


@dataclass(frozen=True)
class TargetDerivedDraftDefaults:
    """Defaults used when a target config generates a draft config.

    ``populate`` is the algorithm-owned seam for derived values such as
    DFlash target-layer IDs.  Reading a config and resolving its model/config
    class remain responsibilities of ``DraftModelRegistry``.
    """

    model_type: str
    num_hidden_layers: int
    draft_vocab_size: int | None = None
    populate: Factory | None = None

    def __post_init__(self) -> None:
        _non_empty(self.model_type, field_name="model_type")
        if (
            isinstance(self.num_hidden_layers, bool)
            or not isinstance(self.num_hidden_layers, int)
            or self.num_hidden_layers <= 0
        ):
            raise ValueError("num_hidden_layers must be a positive integer")
        if self.draft_vocab_size is not None and (
            isinstance(self.draft_vocab_size, bool)
            or not isinstance(self.draft_vocab_size, int)
            or self.draft_vocab_size <= 0
        ):
            raise ValueError("draft_vocab_size must be a positive integer or None")
        if self.populate is not None and not callable(self.populate):
            raise TypeError("populate must be callable or None")


@dataclass(frozen=True)
class DraftConfigProvider:
    """Algorithm policy around a registered draft architecture.

    This object does not contain a model/config class and does not load a
    config.  It supplies algorithm-specific generation and override behavior
    to the composition root after ``DraftModelRegistry`` resolves the class.
    """

    architecture: str
    target_defaults: TargetDerivedDraftDefaults | None = None
    expected_auto_map_model: str | None = None
    apply_overrides: Factory | None = None

    def __post_init__(self) -> None:
        _non_empty(self.architecture, field_name="architecture")
        if self.expected_auto_map_model is not None:
            _non_empty(
                self.expected_auto_map_model,
                field_name="expected_auto_map_model",
            )
        if self.apply_overrides is not None and not callable(self.apply_overrides):
            raise TypeError("apply_overrides must be callable or None")


@dataclass(frozen=True)
class StepRuntimeConfig(Mapping[str, Any]):
    """Resolved step options and checkpoint policy for one training run.

    The mapping interface exposes only strategy constructor options, so the
    existing runtime builders can forward this object through their
    ``strategy_kwargs`` seam. ``Trainer`` additionally reads the immutable
    checkpoint metadata carried alongside those options.
    """

    options: Mapping[str, Any]
    resume_contract: Mapping[str, Any]
    allowed_missing_checkpoint_keys: FrozenSet[str]

    def __post_init__(self) -> None:
        options = dict(self.options)
        invalid_option_keys = sorted(
            repr(key) for key in options if not isinstance(key, str) or not key.strip()
        )
        if invalid_option_keys:
            raise ValueError(
                "step option keys must be non-empty strings, got "
                f"{invalid_option_keys}"
            )
        resume_contract = dict(self.resume_contract)
        invalid_contract_keys = sorted(
            repr(key)
            for key in resume_contract
            if not isinstance(key, str) or not key.strip()
        )
        if invalid_contract_keys:
            raise ValueError(
                "resume_contract keys must be non-empty strings, got "
                f"{invalid_contract_keys}"
            )
        expected_options_contract = tuple(
            (key, options[key]) for key in sorted(options)
        )
        recorded_options_contract = resume_contract.setdefault(
            STEP_OPTIONS_CONTRACT_KEY,
            expected_options_contract,
        )
        if recorded_options_contract != expected_options_contract:
            raise ValueError(
                f"{STEP_OPTIONS_CONTRACT_KEY!r} must exactly match the resolved "
                f"step options: expected {expected_options_contract!r}, got "
                f"{recorded_options_contract!r}"
            )
        allowed_missing = frozenset(self.allowed_missing_checkpoint_keys)
        invalid_missing_keys = sorted(
            repr(key)
            for key in allowed_missing
            if not isinstance(key, str) or not key.strip()
        )
        if invalid_missing_keys:
            raise ValueError(
                "allowed missing checkpoint keys must be non-empty strings, got "
                f"{invalid_missing_keys}"
            )
        if (
            allowed_missing
            and OMITTED_STATE_FINGERPRINT_CONTRACT_KEY not in resume_contract
        ):
            raise ValueError(
                "allowed missing checkpoint keys require the runtime-owned "
                f"{OMITTED_STATE_FINGERPRINT_CONTRACT_KEY!r} resume contract"
            )
        object.__setattr__(self, "options", MappingProxyType(options))
        object.__setattr__(
            self,
            "resume_contract",
            MappingProxyType(resume_contract),
        )
        object.__setattr__(self, "allowed_missing_checkpoint_keys", allowed_missing)

    def __getitem__(self, key: str) -> Any:
        return self.options[key]

    def __iter__(self):
        return iter(self.options)

    def __len__(self) -> int:
        return len(self.options)


@dataclass(frozen=True)
class StepProvider:
    """Factories and persistence hooks for one algorithm's training step."""

    build: Factory
    options: Factory
    resume_contract: Factory
    allowed_missing_checkpoint_keys: Factory
    uses_external_target_head: bool

    def __post_init__(self) -> None:
        _factories_are_callable(
            (
                ("build", self.build),
                ("options", self.options),
                ("resume_contract", self.resume_contract),
                (
                    "allowed_missing_checkpoint_keys",
                    self.allowed_missing_checkpoint_keys,
                ),
            )
        )
        if not isinstance(self.uses_external_target_head, bool):
            raise TypeError("uses_external_target_head must be a bool")

    def bind_runtime(
        self,
        config: Any,
        draft_model: Any,
        training_model: Any,
        *,
        model_provenance: Mapping[str, Any] | None = None,
    ) -> StepRuntimeConfig:
        """Resolve algorithm checkpoint semantics against the live model."""

        options = self.options(config)
        if not isinstance(options, Mapping):
            raise TypeError(
                "step options must return a mapping, got " f"{type(options).__name__}"
            )
        provider_contract = self.resume_contract(config, draft_model, training_model)
        if not isinstance(provider_contract, Mapping):
            raise TypeError(
                "step resume_contract must return a mapping, got "
                f"{type(provider_contract).__name__}"
            )
        resume_contract = dict(provider_contract)
        reserved = {
            STEP_OPTIONS_CONTRACT_KEY,
            MODEL_PROVENANCE_CONTRACT_KEY,
            OMITTED_STATE_FINGERPRINT_CONTRACT_KEY,
        } & resume_contract.keys()
        if reserved:
            raise ValueError(
                "provider resume_contract uses runtime-owned fields: "
                f"{sorted(reserved)}"
            )
        resume_contract[STEP_OPTIONS_CONTRACT_KEY] = tuple(
            (key, options[key]) for key in sorted(options)
        )
        if model_provenance is not None:
            resume_contract[MODEL_PROVENANCE_CONTRACT_KEY] = tuple(
                (key, model_provenance[key]) for key in sorted(model_provenance)
            )
        allowed_missing = frozenset(
            self.allowed_missing_checkpoint_keys(
                config,
                draft_model,
                training_model,
            )
        )
        if allowed_missing:
            resume_contract[OMITTED_STATE_FINGERPRINT_CONTRACT_KEY] = (
                checkpoint_key_fingerprint(draft_model, allowed_missing)
            )
        return StepRuntimeConfig(
            options=options,
            resume_contract=resume_contract,
            allowed_missing_checkpoint_keys=allowed_missing,
        )


@dataclass(frozen=True)
class ModelProvider:
    """Algorithm-owned model assembly hooks.

    Keeping concrete model imports inside hook functions lets the immutable
    registry resolve without importing Torch or Transformers.
    """

    draft_config: DraftConfigProvider
    build_draft: Factory
    build_training_model: Factory
    resolve_capture_layers: Factory
    minimum_loss_tokens: Factory
    needs_input_tools: Factory
    default_dataloader_num_workers: int
    allow_missing_warm_start_embedding: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.draft_config, DraftConfigProvider):
            raise TypeError("draft_config must be a DraftConfigProvider")
        _factories_are_callable(
            (
                ("build_draft", self.build_draft),
                ("build_training_model", self.build_training_model),
                ("resolve_capture_layers", self.resolve_capture_layers),
                ("minimum_loss_tokens", self.minimum_loss_tokens),
                ("needs_input_tools", self.needs_input_tools),
            )
        )
        if (
            isinstance(self.default_dataloader_num_workers, bool)
            or not isinstance(self.default_dataloader_num_workers, int)
            or self.default_dataloader_num_workers < 0
        ):
            raise ValueError(
                "default_dataloader_num_workers must be a non-negative integer"
            )
        if not isinstance(self.allow_missing_warm_start_embedding, bool):
            raise TypeError("allow_missing_warm_start_embedding must be a bool")


@dataclass(frozen=True)
class OfflineCaptureLayout:
    """Map one generic local capture onto the persisted offline schema.

    Local capture always exposes the same four per-sample sources:
    ``input_ids``, ``loss_mask``, ``aux_hidden_states``, and
    ``last_hidden_states``. Algorithms own the names they persist so the data
    preparation command cannot silently drift away from their storage
    contracts.
    """

    capture_method: str
    aux_feature: str | None
    last_hidden_feature: str | None
    passthrough: Tuple[Tuple[str, str], ...]

    def __post_init__(self) -> None:
        _non_empty(self.capture_method, field_name="capture_method")
        for field_name in ("aux_feature", "last_hidden_feature"):
            value = getattr(self, field_name)
            if value is not None:
                _non_empty(value, field_name=field_name)

        passthrough = tuple(self.passthrough)
        for feature_name, source_key in passthrough:
            _non_empty(feature_name, field_name="passthrough feature name")
            _non_empty(source_key, field_name="passthrough source key")
        object.__setattr__(self, "passthrough", passthrough)

        output_names = self.output_names
        duplicates = sorted(
            name for name in set(output_names) if output_names.count(name) > 1
        )
        if duplicates:
            raise ValueError(
                f"duplicate offline capture output feature names: {duplicates}"
            )
        if not output_names:
            raise ValueError("offline capture layout must emit at least one feature")

    @property
    def output_names(self) -> Tuple[str, ...]:
        """Return persisted feature names in deterministic layout order."""

        names = [feature for feature, _source in self.passthrough]
        if self.aux_feature is not None:
            names.append(self.aux_feature)
        if self.last_hidden_feature is not None:
            names.append(self.last_hidden_feature)
        return tuple(names)

    def materialize(self, sources: Mapping[str, Any]) -> dict[str, Any]:
        """Build one persisted record and fail early on missing capture data."""

        mappings = list(self.passthrough)
        if self.aux_feature is not None:
            mappings.append((self.aux_feature, "aux_hidden_states"))
        if self.last_hidden_feature is not None:
            mappings.append((self.last_hidden_feature, "last_hidden_states"))

        record = {}
        for feature_name, source_key in mappings:
            if source_key not in sources:
                raise KeyError(
                    f"offline capture source {source_key!r} required for "
                    f"feature {feature_name!r} is missing"
                )
            value = sources[source_key]
            if value is None:
                raise ValueError(
                    f"offline capture source {source_key!r} required for "
                    f"feature {feature_name!r} is None"
                )
            record[feature_name] = value
        return record


@dataclass(frozen=True)
class OfflineDataProvider:
    """Reader, normalizer, collator, and capture schema for one modality."""

    modality: str
    normalizer_id: str
    build_reader: Factory
    build_normalizer: Factory
    build_collator: Factory
    capture_layout: OfflineCaptureLayout | None = None

    def __post_init__(self) -> None:
        _non_empty(self.modality, field_name="modality")
        _non_empty(self.normalizer_id, field_name="normalizer_id")
        if self.capture_layout is not None and not isinstance(
            self.capture_layout,
            OfflineCaptureLayout,
        ):
            raise TypeError("capture_layout must be an OfflineCaptureLayout or None")
        _factories_are_callable(
            (
                ("build_reader", self.build_reader),
                ("build_normalizer", self.build_normalizer),
                ("build_collator", self.build_collator),
            )
        )


@dataclass(frozen=True)
class ServerCaptureLayout:
    """Maps generic server artifacts onto algorithm-ready feature names."""

    aux_feature: str | None
    last_hidden_feature: str | None
    passthrough: Tuple[Tuple[str, str, Tuple[int, ...]], ...]
    attention_mask_feature: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "aux_feature",
            "last_hidden_feature",
            "attention_mask_feature",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _non_empty(value, field_name=field_name)
        passthrough = tuple(self.passthrough)
        for feature_name, payload_key, trailing_shape in passthrough:
            _non_empty(feature_name, field_name="passthrough feature name")
            _non_empty(payload_key, field_name="passthrough payload key")
            if any(
                isinstance(size, bool) or not isinstance(size, int) or size < 0
                for size in trailing_shape
            ):
                raise ValueError(
                    "passthrough trailing shapes must contain non-negative integers"
                )
        object.__setattr__(self, "passthrough", passthrough)


@dataclass(frozen=True)
class ServerStreamingProvider:
    """Algorithm adapter for externally captured streaming features.

    ``build_input_adapter`` is deliberately modality-neutral. Text providers
    can leave it unset; the current runtime does not support VLM registration
    or media requests.
    """

    modality: str
    capture_method: str
    target_representation: str | None
    layout: ServerCaptureLayout
    build_collator: Factory
    build_input_adapter: Factory | None = None

    def __post_init__(self) -> None:
        _non_empty(self.modality, field_name="modality")
        _non_empty(self.capture_method, field_name="capture_method")
        if self.target_representation is not None:
            _non_empty(
                self.target_representation,
                field_name="target_representation",
            )
        if not isinstance(self.layout, ServerCaptureLayout):
            raise TypeError("layout must be a ServerCaptureLayout")
        if not callable(self.build_collator):
            raise TypeError("build_collator must be callable")
        if self.build_input_adapter is not None and not callable(
            self.build_input_adapter
        ):
            raise TypeError("build_input_adapter must be callable or None")

    def create_input_adapter(self, config: Any) -> ServerInputAdapter | None:
        """Construct and validate the optional modality-owned input adapter."""

        if self.build_input_adapter is None:
            return None
        adapter = self.build_input_adapter(config)
        required = (
            "load_input_tools",
            "prepare_prompts",
            "build_request_inputs",
        )
        missing = [
            name for name in required if not callable(getattr(adapter, name, None))
        ]
        if missing:
            raise TypeError(
                "build_input_adapter must return a ServerInputAdapter; "
                f"missing callable methods: {missing}"
            )
        return adapter


@dataclass(frozen=True)
class AlgorithmProviders:
    """Executable providers paired with one pure ``AlgorithmSpec``."""

    algorithm_name: str
    step: StepProvider
    model: ModelProvider
    offline: Tuple[OfflineDataProvider, ...] = ()
    server_streaming: Tuple[ServerStreamingProvider, ...] = ()
    vocab_mapping_modes: FrozenSet[FeatureMode] = frozenset()

    def __post_init__(self) -> None:
        _non_empty(self.algorithm_name, field_name="algorithm_name")
        if not isinstance(self.step, StepProvider):
            raise TypeError("step must be a StepProvider")
        if not isinstance(self.model, ModelProvider):
            raise TypeError("model must be a ModelProvider")
        offline = tuple(self.offline)
        streaming = tuple(self.server_streaming)
        self._validate_unique_modalities(offline, field_name="offline")
        self._validate_unique_modalities(
            streaming,
            field_name="server_streaming",
        )
        modes = frozenset(FeatureMode(mode) for mode in self.vocab_mapping_modes)
        object.__setattr__(self, "offline", offline)
        object.__setattr__(self, "server_streaming", streaming)
        object.__setattr__(self, "vocab_mapping_modes", modes)

    @staticmethod
    def _validate_unique_modalities(providers, *, field_name: str) -> None:
        modalities = [provider.modality for provider in providers]
        duplicates = sorted(
            modality for modality in set(modalities) if modalities.count(modality) > 1
        )
        if duplicates:
            raise ValueError(f"duplicate {field_name} modalities: {duplicates}")

    def offline_for(self, modality: str) -> OfflineDataProvider:
        return self._provider_for(self.offline, modality, kind="offline")

    def server_streaming_for(self, modality: str) -> ServerStreamingProvider:
        return self._provider_for(
            self.server_streaming,
            modality,
            kind="server streaming",
        )

    def _provider_for(self, providers, modality: str, *, kind: str):
        for provider in providers:
            if provider.modality == modality:
                return provider
        available = sorted(provider.modality for provider in providers)
        raise KeyError(
            f"algorithm {self.algorithm_name!r} has no {kind} provider for "
            f"modality {modality!r}; available: {available}"
        )


def make_registration(
    spec: AlgorithmSpec,
    providers: AlgorithmProviders,
) -> AlgorithmRegistration:
    """Validate contract/provider parity and return one registration."""

    if providers.algorithm_name != spec.name:
        raise ValueError(
            f"provider algorithm {providers.algorithm_name!r} does not match "
            f"spec {spec.name!r}"
        )
    if providers.model.draft_config.architecture not in (
        spec.draft.compatible_architectures
    ):
        raise ValueError(
            "model provider architecture is not compatible with the algorithm "
            f"contract: {providers.model.draft_config.architecture!r}"
        )

    expected = {contract.key for contract in spec.feature_contracts}
    provided = {
        *((FeatureMode.OFFLINE, provider.modality) for provider in providers.offline),
        *(
            (FeatureMode.STREAMING, provider.modality)
            for provider in providers.server_streaming
        ),
    }
    if provided != expected:
        raise ValueError(
            "feature contract/provider keys differ: "
            f"contracts={sorted((mode.value, modality) for mode, modality in expected)}, "
            f"providers={sorted((mode.value, modality) for mode, modality in provided)}"
        )
    has_vocab_mapping_provider = bool(providers.vocab_mapping_modes)
    if has_vocab_mapping_provider != spec.capabilities.supports_vocab_mapping:
        raise ValueError(
            "vocab-mapping capability/provider mismatch: "
            f"contract={spec.capabilities.supports_vocab_mapping}, "
            f"provider_modes={sorted(mode.value for mode in providers.vocab_mapping_modes)}"
        )
    unavailable_mapping_modes = providers.vocab_mapping_modes - spec.feature_modes
    if unavailable_mapping_modes:
        raise ValueError(
            "vocab-mapping provider modes lack feature contracts: "
            f"{sorted(mode.value for mode in unavailable_mapping_modes)}"
        )

    for provider in providers.offline:
        contract = spec.feature_contract(FeatureMode.OFFLINE, provider.modality)
        if contract.storage is None:  # pragma: no cover - contract invariant
            raise ValueError("offline contract is missing storage")
        if provider.normalizer_id != contract.storage.normalizer:
            raise ValueError(
                f"offline normalizer mismatch for {provider.modality!r}: "
                f"{provider.normalizer_id!r} != {contract.storage.normalizer!r}"
            )
        if provider.capture_layout is not None:
            emitted = set(provider.capture_layout.output_names)
            required = set(contract.storage.required_tensors)
            if emitted != required:
                raise ValueError(
                    f"offline capture layout for {provider.modality!r} must "
                    "exactly match the storage contract: "
                    f"emitted={sorted(emitted)}, required={sorted(required)}"
                )
            streaming = next(
                (
                    candidate
                    for candidate in providers.server_streaming
                    if candidate.modality == provider.modality
                ),
                None,
            )
            if (
                streaming is not None
                and provider.capture_layout.capture_method != streaming.capture_method
            ):
                raise ValueError(
                    f"offline and server capture methods differ for "
                    f"{provider.modality!r}: "
                    f"{provider.capture_layout.capture_method!r} != "
                    f"{streaming.capture_method!r}"
                )
    for provider in providers.server_streaming:
        contract = spec.feature_contract(FeatureMode.STREAMING, provider.modality)
        if provider.target_representation != contract.default_target_representation:
            raise ValueError(
                f"streaming target representation mismatch for "
                f"{provider.modality!r}: {provider.target_representation!r} != "
                f"{contract.default_target_representation!r}"
            )
        layout = provider.layout
        emitted = {
            *(
                feature
                for feature in (
                    layout.aux_feature,
                    layout.last_hidden_feature,
                    layout.attention_mask_feature,
                )
                if feature is not None
            ),
            *(feature for feature, _payload, _shape in layout.passthrough),
        }
        missing = contract.required_tensors - emitted
        if missing:
            raise ValueError(
                f"server capture layout for {provider.modality!r} does not emit "
                f"required tensors: {sorted(missing)}"
            )

    return AlgorithmRegistration(spec=spec, providers=providers)


__all__ = [
    "AlgorithmProviders",
    "DraftConfigProvider",
    "MODEL_PROVENANCE_CONTRACT_KEY",
    "ModelProvider",
    "OMITTED_STATE_FINGERPRINT_CONTRACT_KEY",
    "OfflineCaptureLayout",
    "OfflineDataProvider",
    "STEP_OPTIONS_CONTRACT_KEY",
    "ServerCaptureLayout",
    "ServerInputAdapter",
    "ServerStreamingProvider",
    "StepProvider",
    "StepRuntimeConfig",
    "TargetDerivedDraftDefaults",
    "checkpoint_key_fingerprint",
    "make_registration",
]
