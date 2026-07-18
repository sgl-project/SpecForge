"""Algorithm-aware validation between the config and executable layers."""

from __future__ import annotations

from specforge.algorithms.contracts import FeatureMode
from specforge.algorithms.registry import AlgorithmRegistration
from specforge.config import Config


def _feature_mode(cfg: Config) -> FeatureMode:
    return FeatureMode.OFFLINE if cfg.mode == "offline" else FeatureMode.STREAMING


def _validate_feature_provider(
    cfg: Config,
    algorithm: AlgorithmRegistration,
    mode: FeatureMode,
) -> None:
    modality = cfg.model.input_modality
    spec = algorithm.spec
    if not spec.supports(mode, modality):
        supported = sorted(
            (contract.mode.value, contract.modality)
            for contract in spec.feature_contracts
        )
        raise ValueError(
            f"algorithm {algorithm.name!r} has no {mode.value} feature contract "
            f"and provider for modality {modality!r}; supported: {supported}"
        )

    # Registration construction enforces contract/provider parity. Resolve the
    # provider here as a defensive assertion and to keep modality failures at
    # this generic application boundary.
    if mode is FeatureMode.OFFLINE:
        algorithm.providers.offline_for(modality)
    else:
        algorithm.providers.server_streaming_for(modality)


def _validate_draft_options(
    cfg: Config,
    algorithm: AlgorithmRegistration,
) -> None:
    requirement = algorithm.spec.draft
    provider = algorithm.providers.model.draft_config
    if (
        not cfg.model.draft_model_config
        and not cfg.model.draft_checkpoint_path
        and provider.target_defaults is None
    ):
        raise ValueError(
            f"training.strategy={algorithm.name!r} requires "
            "model.draft_model_config; automatic target-derived configs are "
            "not registered for this algorithm"
        )

    overrides = {
        "num_hidden_layers": cfg.model.draft_num_hidden_layers,
        "block_size": cfg.model.draft_block_size,
    }
    fixed_values = dict(requirement.fixed_override_values)
    for name, value in overrides.items():
        if value is not None and name not in requirement.supported_overrides:
            raise ValueError(
                f"algorithm {algorithm.name!r} does not support " f"model.draft_{name}"
            )
        if value is not None and name in fixed_values and value != fixed_values[name]:
            raise ValueError(
                f"algorithm {algorithm.name!r} requires model.draft_{name}="
                f"{fixed_values[name]} or no override"
            )


def _validate_algorithm_capabilities(
    cfg: Config,
    algorithm: AlgorithmRegistration,
    mode: FeatureMode,
) -> None:
    capabilities = algorithm.spec.capabilities
    training = cfg.training
    if training.attention_backend not in capabilities.attention_backends:
        raise ValueError(
            f"algorithm {algorithm.name!r} does not support attention_backend="
            f"{training.attention_backend!r}; supported: "
            f"{sorted(capabilities.attention_backends)}"
        )
    if (
        capabilities.required_batch_size is not None
        and training.batch_size != capabilities.required_batch_size
    ):
        raise ValueError(
            f"algorithm {algorithm.name!r} requires training.batch_size="
            f"{capabilities.required_batch_size}"
        )

    layers = cfg.model.aux_hidden_state_layer_ids
    if layers is not None:
        if not capabilities.allows_aux_layer_override:
            raise ValueError(
                f"algorithm {algorithm.name!r} gets capture layers from its draft "
                "config; model.aux_hidden_state_layer_ids would be ignored"
            )
        if (
            len(layers) != 3
            or any(isinstance(layer, bool) or layer < 0 for layer in layers)
            or len(set(layers)) != len(layers)
        ):
            raise ValueError(
                "model.aux_hidden_state_layer_ids must contain exactly three "
                "distinct non-negative layer ids"
            )

    if training.compact_teacher and (
        not capabilities.supports_compact_teacher
        or mode is not FeatureMode.OFFLINE
        or cfg.model.input_modality != "text"
    ):
        raise ValueError(
            f"algorithm {algorithm.name!r} does not support compact teacher for "
            f"mode={mode.value!r}, modality={cfg.model.input_modality!r}"
        )


def _validate_training_topology(
    cfg: Config,
    mode: FeatureMode,
) -> None:
    deployment_mode = cfg.deployment.mode
    if mode is FeatureMode.OFFLINE and cfg.training.tp_size != 1:
        raise ValueError(
            "offline feature consumers do not implement trainer tensor "
            "parallelism; keep training.tp_size=1 so every non-SP rank "
            "receives its own data shard"
        )
    if mode is FeatureMode.STREAMING:
        if deployment_mode != "disaggregated":
            raise ValueError(
                "online training requires deployment.mode=disaggregated; "
                "colocated online training is no longer supported"
            )
        if cfg.model.target_backend != "sglang":
            raise ValueError(
                "online training uses an external SGLang capture server and "
                "requires model.target_backend=sglang"
            )
        deployment = cfg.deployment.disaggregated
        if deployment is None or deployment.backend != "mooncake":
            raise ValueError(
                "online disaggregated training requires "
                "deployment.disaggregated.backend=mooncake"
            )
        if cfg.model.shard_target_output:
            raise ValueError(
                "model.shard_target_output is unavailable with external server "
                "capture"
            )
        if (
            cfg.training.tp_size != 1
            or cfg.training.sp_ulysses_size != 1
            or cfg.training.sp_ring_size != 1
        ):
            raise ValueError(
                "the disaggregated online consumer uses every trainer rank for "
                "data parallelism; configure target TP on the external server and "
                "keep training.tp_size/sp sizes at 1"
            )

    if cfg.training.attention_backend == "usp" and mode is not FeatureMode.OFFLINE:
        raise ValueError("USP attention currently requires offline features")


def _validate_vocab_mapping(
    cfg: Config,
    algorithm: AlgorithmRegistration,
    mode: FeatureMode,
) -> None:
    if (
        cfg.deployment.mode == "disaggregated"
        and mode in algorithm.providers.vocab_mapping_modes
        and not cfg.model.vocab_mapping_path
    ):
        raise ValueError(
            f"algorithm {algorithm.name!r} disaggregated runs require "
            "model.vocab_mapping_path because producer and consumer cannot "
            "derive one shared mapping"
        )


def validate_resolved_run(
    cfg: Config,
    algorithm: AlgorithmRegistration,
) -> None:
    """Validate one config against its resolved pure contract and providers."""

    if algorithm.name != cfg.training.strategy:
        raise ValueError(
            "resolved algorithm does not match training.strategy: "
            f"{algorithm.name!r} != {cfg.training.strategy!r}"
        )
    mode = _feature_mode(cfg)
    _validate_feature_provider(cfg, algorithm, mode)
    _validate_draft_options(cfg, algorithm)
    _validate_algorithm_capabilities(cfg, algorithm, mode)
    _validate_training_topology(cfg, mode)
    _validate_vocab_mapping(cfg, algorithm, mode)


__all__ = ["validate_resolved_run"]
