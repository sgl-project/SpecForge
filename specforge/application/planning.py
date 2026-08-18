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

    if training.trim_loss_positions and not capabilities.supports_trim_loss_positions:
        raise ValueError(
            f"algorithm {algorithm.name!r} does not support "
            "training.trim_loss_positions"
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


def _prunes_vocabulary(
    cfg: Config,
    algorithm: AlgorithmRegistration,
) -> Optional[bool]:
    """Whether this run's draft actually predicts over a pruned vocabulary.

    Declaring ``supports_vocab_mapping`` says the algorithm *can* prune; only
    ``draft_vocab_size < vocab_size`` in the resolved draft config says this run
    *does*. Mapping requirements must key off the latter, or turning the
    capability on would start rejecting full-vocabulary configs that never
    needed a mapping.

    Returns ``None`` when the draft config cannot be read. Resolving it is not
    this check's job -- config validation must keep working without the draft
    config on disk -- so each caller decides what an unknown answer means for
    the rule it enforces, rather than this guessing an answer for all of them.
    """
    if not algorithm.spec.capabilities.supports_vocab_mapping:
        return False

    from specforge.training.model_loading import draft_config_dict

    try:
        draft_cfg = draft_config_dict(
            cfg, provider=algorithm.providers.model.draft_config
        )
    except Exception:
        return None
    vocab_size = draft_cfg.get("vocab_size")
    draft_vocab_size = draft_cfg.get("draft_vocab_size") or vocab_size
    if vocab_size is None or draft_vocab_size is None:
        return False
    return int(draft_vocab_size) != int(vocab_size)


def _validate_vocab_mapping(
    cfg: Config,
    algorithm: AlgorithmRegistration,
    mode: FeatureMode,
) -> None:
    supports_mapping = algorithm.spec.capabilities.supports_vocab_mapping
    if cfg.model.vocab_mapping_path and not supports_mapping:
        raise ValueError(
            f"algorithm {algorithm.name!r} does not support vocabulary mapping, "
            "so model.vocab_mapping_path would be silently ignored; remove it"
        )
    # Supporting mapping is not the same as being able to consume one. A draft
    # that only registers t2d/d2t when it prunes cannot load a mapping at full
    # vocabulary, and without this the path is accepted here and then fails much
    # later with a "t2d/d2t buffers are not present" error that points at the
    # model instead of at the config. Drafts that always carry the buffers just
    # install an identity map, which is redundant but works, so they are left
    # alone rather than having a shipped config invalidated retroactively.
    # An unknown answer is not evidence of a full vocabulary, so this rejects
    # only a config proven to be unpruned; the unreadable draft config then
    # surfaces at the model-loading boundary with its own error.
    if (
        cfg.model.vocab_mapping_path
        and supports_mapping
        and not algorithm.spec.capabilities.keeps_vocab_buffers_when_unpruned
        and _prunes_vocabulary(cfg, algorithm) is False
    ):
        raise ValueError(
            f"algorithm {algorithm.name!r} run has draft_vocab_size == "
            "vocab_size, so its draft carries no t2d/d2t buffers for "
            "model.vocab_mapping_path to load into; set a smaller "
            "draft_vocab_size in the draft config or remove the path"
        )
    # An unknown answer falls back to what the algorithm needs by default: a
    # draft that always carries t2d/d2t always needs the two sides to agree on
    # one mapping, so an unreadable config must not drop a requirement that held
    # before pruning was configurable. A draft that carries them only when it
    # prunes has nothing to share until the config says it prunes.
    prunes = _prunes_vocabulary(cfg, algorithm)
    if prunes is None:
        prunes = algorithm.spec.capabilities.keeps_vocab_buffers_when_unpruned
    if (
        cfg.deployment.mode == "disaggregated"
        and mode in algorithm.providers.vocab_mapping_modes
        and not cfg.model.vocab_mapping_path
        and prunes
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
