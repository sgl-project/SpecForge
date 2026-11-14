"""Compose canonical models/objectives with existing data and trainer ports."""

import logging
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path


def enabled(cfg):
    if (
        cfg.draft is not None
        or cfg.initialization is not None
        or cfg.resume is not None
    ):
        return True
    if cfg.resume_path:
        from specforge.checkpoint.resolve import read_json
        from specforge.training.checkpoint import CheckpointManager

        try:
            directory = CheckpointManager.resolve_resume_dir(cfg.resume_path)
        except FileNotFoundError:
            return False  # Legacy launch planning need not have a local checkpoint.
        path = Path(directory) / "config.json"
        return path.is_file() and read_json(path).get("model_type") == "specforge_draft"
    return False


_LEGACY_SETTING_FIELDS = {
    "mask_token_id": "model.mask_token_id",
    "norm_before_residual": "training.norm_before_residual",
    "capture_layers": "model.aux_hidden_state_layer_ids",
}


def _legacy_settings(cfg):
    return {
        name: value
        for name, value in {
            "mask_token_id": cfg.model.mask_token_id,
            "norm_before_residual": cfg.training.norm_before_residual,
            "capture_layers": cfg.model.aux_hidden_state_layer_ids,
        }.items()
        if value is not None
    }


def _is_canonical_checkpoint(cfg, source):
    from specforge.checkpoint.resolve import read_json, resolve_checkpoint_path

    path = resolve_checkpoint_path(
        source, cache_dir=cfg.model.cache_dir, config_only=True
    )
    config = (path.parent if path.is_file() else path) / "config.json"
    return config.is_file() and read_json(config).get("model_type") == "specforge_draft"


def _source_legacy_settings(cfg, source):
    """Legacy imports apply the YAML settings; canonical sources already record them."""
    return None if _is_canonical_checkpoint(cfg, source) else _legacy_settings(cfg)


def _reject_changed_legacy_settings(config, settings):
    spec = config.spec_config
    features = spec.target_features
    # each setting lives on the spec node of the architectures that have it
    recorded = {
        "mask_token_id": getattr(
            spec, "mask_token_id", getattr(features, "mask_token_id", None)
        ),
        "norm_before_residual": getattr(spec, "norm_before_residual", None),
        "capture_layers": getattr(features, "target_layer_ids", None),
    }
    for name, value in settings.items():
        saved = recorded[name]
        if name == "capture_layers" and saved is not None:
            saved, value = list(saved), list(value)
        if saved != value:
            raise ValueError(
                f"{_LEGACY_SETTING_FIELDS[name]}={value!r} differs from the "
                f"checkpoint's recorded {saved!r}; remove it or make it match"
            )


def resolve_config(cfg, *, target_config=None):
    from specforge.config.draft.codec import parse_draft

    draft = cfg.draft
    if isinstance(draft, str):
        import json

        import yaml

        text = Path(draft).read_text()
        try:
            draft = json.loads(text)
        except json.JSONDecodeError:
            draft = yaml.safe_load(text)
    legacy_draft = isinstance(draft, dict) and draft.get("model_type") not in (
        None,
        "specforge_draft",
    )
    if legacy_draft:
        from specforge.config.draft.codec import decode_legacy_config

        if target_config is None:
            from specforge.checkpoint.resolve import read_target_config

            target_config = read_target_config(
                cfg.model.target_model_path, cache_dir=cfg.model.cache_dir
            )
        supplied = decode_legacy_config(
            draft, target_config=target_config, **_legacy_settings(cfg)
        )
    else:
        supplied = (
            parse_draft(draft, target_config=target_config)
            if draft is not None
            else None
        )
    initialization = cfg.initialization
    source = initialization.checkpoint if initialization else None
    if cfg.resume_path:
        from specforge.training.checkpoint import CheckpointManager

        source = CheckpointManager.resolve_resume_dir(cfg.resume_path)
    if source:
        from specforge.loading import read_draft

        canonical = _is_canonical_checkpoint(cfg, source)
        if cfg.resume_path and not canonical:
            raise ValueError(
                "legacy full resume uses the legacy training path; use initialization.checkpoint for canonical weight import"
            )
        loaded = read_draft(
            source,
            target=cfg.model.target_model_path,
            cache_dir=cfg.model.cache_dir,
            adapter=initialization.adapter if initialization else None,
            legacy_config=initialization.legacy_config if initialization else None,
            legacy_settings=None if canonical else _legacy_settings(cfg),
        ).config
        if canonical:
            _reject_changed_legacy_settings(loaded, _legacy_settings(cfg))
        if supplied is not None and supplied != loaded:
            raise ValueError(
                "draft must exactly match checkpoint structure; use weights_from to change it"
            )
        return loaded
    if (
        _legacy_settings(cfg)
        and not legacy_draft
        and not (initialization and initialization.weights_from)
    ):
        raise ValueError(
            "canonical draft settings belong in draft; legacy overrides require a legacy checkpoint"
        )
    if supplied is None:
        raise ValueError("canonical training requires draft or a checkpoint")
    return supplied


def capture_layers(cfg, draft, target_config):
    from specforge.config.draft.codec import parse_draft

    if isinstance(draft, dict):
        draft = parse_draft(
            draft,
            target_config=target_config,
        )
    from specforge.config.draft.architecture.features import FinalHiddenFeatureSpec

    features = draft.spec_config.target_features
    if isinstance(features, FinalHiddenFeatureSpec):
        if target_config["num_hidden_layers"] < 2:
            raise ValueError(
                "final-hidden capture requires a target with at least two layers"
            )
        # DFLASH capture requires an auxiliary tap; MTP discards this feature.
        return [0]
    return list(features.target_layer_ids)


def minimum_loss_tokens(cfg, draft, *, count):
    return count


def build_step(wrapped_model, *, target_head=None, **options):
    from specforge.training.strategies.canonical import CanonicalTrainStrategy

    return CanonicalTrainStrategy(wrapped_model, **options)


def _objective_settings(cfg):
    return dict(
        name=cfg.training.objective,
        options=cfg.training.objective_options,
        loss=cfg.training.loss,
        weighting=cfg.training.weighting,
        execution=cfg.training.loss_execution,
    )


def _validate_imported_objective(cfg, resolved):
    init = cfg.initialization
    if init is None or init.checkpoint is None:
        return
    from specforge.checkpoint.resolve import read_json, resolve_directory
    from specforge.config.draft.codec.legacy.dspine import (
        extract_dspine_training_settings,
    )

    source = resolve_directory(
        init.checkpoint, cache_dir=cfg.model.cache_dir, config_only=True
    )
    directory = source.parent if source.is_file() else source
    payload = read_json(init.legacy_config or directory / "config.json")
    if payload.get("model_type") == "specforge_draft" or payload.get(
        "architectures"
    ) != ["DSpineDraftModel"]:
        return
    for name, saved in extract_dspine_training_settings(payload).items():
        if getattr(resolved.objective, name) != saved:
            raise ValueError(
                f"legacy DSpine {name}={saved!r}; set training.objective_options.{name} to preserve it"
            )


def resolve_run(cfg, *, target_config=None):
    """Reuse deployment/data ports while resolving structure and loss separately."""
    from specforge.algorithms.common.defaults import (
        empty_options,
        empty_resume_contract,
        no_missing_checkpoint_keys,
        online_needs_input_tools,
    )
    from specforge.algorithms.common.providers import (
        AlgorithmProviders,
        DraftConfigProvider,
        ModelProvider,
        OfflineCaptureLayout,
        OfflineDataProvider,
        ServerCaptureLayout,
        ServerStreamingProvider,
        StepProvider,
        make_registration,
    )
    from specforge.algorithms.contracts import (
        AlgorithmCapabilities,
        AlgorithmSpec,
        DraftRequirement,
        FeatureContract,
        FeatureMode,
        OfflineStorageContract,
    )
    from specforge.algorithms.objectives import resolve_objective
    from specforge.config.draft.codec import encode_config

    from . import canonical_data

    if cfg.model.draft_model_config or cfg.model.draft_checkpoint_path:
        raise ValueError(
            "canonical checkpoint cannot mix with legacy model.draft_* sources"
        )
    draft = resolve_config(cfg, target_config=target_config)
    if cfg.initialization and cfg.initialization.from_target:
        from specforge.checkpoint.native import validate_target_initialization
        from specforge.checkpoint.resolve import read_target_config

        target = (
            target_config
            if target_config is not None
            else read_target_config(
                cfg.model.target_model_path, cache_dir=cfg.model.cache_dir
            )
        )
        validate_target_initialization(draft, target, cfg.initialization.from_target)
    # New objectives have their own options. Never silently consume legacy
    # algorithm switches with a different meaning on the canonical path.
    optimization_options = {
        "objective_chunk_blocks",
        "objective_checkpointing",
        "fused_logits_loss",
        "mtp_objective_chunk_size",
        "trim_loss_positions",
    }
    legacy_options = optimization_options | {
        "loss_type",
        "lk_loss_type",
        "dpace_alpha",
        "kl_scale",
        "kl_decay",
        "ttt_length",
        "num_anchors",
        "loss_decay_gamma",
        "dflash2_selector_loss_alpha",
        "dflash2_selector_warmup_ratio",
        "dflash2_selector_ramp_ratio",
        "dflash2_selector_stop_gradient",
        "lambda_base_start",
        "lambda_base_decay_ratio",
        "dspark_ce_loss_alpha",
        "dspark_l1_loss_alpha",
        "dspark_confidence_head_alpha",
        "num_depths",
        "down_sample_ratio",
        "down_sample_ratio_min",
    }
    from specforge.config.schema import TrainingConfig

    defaults = TrainingConfig()
    if draft.architecture == "MTPDraftModel":
        legacy_options.remove("mtp_objective_chunk_size")
        optimization_options.discard("mtp_objective_chunk_size")
        chunk = cfg.training.mtp_objective_chunk_size
        if chunk != defaults.mtp_objective_chunk_size:
            options = dict(cfg.training.objective_options)
            if (
                "objective_chunk_size" in options
                and options["objective_chunk_size"] != chunk
            ):
                raise ValueError("conflicting MTP objective_chunk_size settings")
            options["objective_chunk_size"] = chunk
            cfg = cfg.model_copy(
                update={
                    "training": cfg.training.model_copy(
                        update={
                            "objective_options": options,
                            "mtp_objective_chunk_size": defaults.mtp_objective_chunk_size,
                        }
                    )
                }
            )
    present = {
        name
        for name in legacy_options
        if getattr(cfg.training, name) != getattr(defaults, name)
    }
    if present:
        raise ValueError(
            f"canonical training uses training.loss, training.weighting and "
            f"training.objective_options, not {sorted(present)}"
        )
    ignored = cfg.training.model_fields_set & optimization_options
    if ignored:
        logging.getLogger(__name__).warning(
            "Canonical objectives do not use legacy optimization options %s; "
            "their default values do not enable chunking, checkpointing or fused loss",
            sorted(ignored),
        )
    resolved = resolve_objective(draft.architecture, **_objective_settings(cfg))
    _validate_imported_objective(cfg, resolved)
    if resolved.execution.teacher_chunk_size and cfg.mode != "offline":
        raise ValueError("compact teacher currently requires offline training")
    from specforge.config.schema import ModelConfig

    init = cfg.initialization
    scratch = not (init and (init.checkpoint or init.weights_from))
    used_keys = {
        "lm_head_key": "lm_head" in draft.target_bindings or resolved.requires_teacher,
        "embedding_key": "embed_tokens" in draft.target_bindings
        or resolved.requires_teacher
        or (scratch and cfg.model.load_target_embedding),
    }
    # on resume the recorded model provenance pins both keys instead
    for field, used in () if cfg.resume_path else used_keys.items():
        if (
            not used
            and getattr(cfg.model, field) != ModelConfig.model_fields[field].default
        ):
            raise ValueError(f"model.{field} is not used by this draft initialization")
    objective, entry = resolved.name, resolved.registration
    requirements = entry.training
    family = entry.legacy_strategy or objective
    if cfg.training.strategy not in (defaults.strategy, family, objective):
        raise ValueError("training.strategy conflicts with the objective")
    if (
        cfg.training.tp_size,
        cfg.training.sp_ulysses_size,
        cfg.training.sp_ring_size,
    ) != (1, 1, 1):
        raise ValueError(
            "canonical objectives currently require trainer TP/SP sizes of one"
        )
    # Retain the established strategy name for data/rendezvous identifiers.
    training = cfg.training.model_copy(
        update={"strategy": family, "objective": objective}
    )
    cfg = cfg.model_copy(update={"draft": encode_config(draft), "training": training})
    keys = tuple(entry.inputs)
    normalizer = "canonical_unshifted_v1"
    storage = OfflineStorageContract(
        format="specforge_hidden_states_v1",
        normalizer=normalizer,
        required_tensors=frozenset(keys),
    )
    target_repr = "hidden_state" if "target_last_hidden_states" in keys else None
    representations = frozenset({target_repr}) if target_repr else frozenset()
    contracts = [
        FeatureContract(
            mode=FeatureMode.OFFLINE,
            modality="text",
            required_tensors=frozenset(keys),
            optional_tensors=frozenset({"attention_mask"}),
            allowed_target_representations=representations,
            default_target_representation=target_repr,
            storage=storage,
        )
    ]
    # Generic capture supplies aux/final target states. DMTP needs an external
    # frozen-MTP feature producer, so only precomputed records are accepted.
    layout = (
        None
        if requirements.capture_method is None
        else OfflineCaptureLayout(
            capture_method=requirements.capture_method,
            aux_feature="hidden_states" if "hidden_states" in keys else None,
            last_hidden_feature="target_last_hidden_states" if target_repr else None,
            passthrough=(("input_ids", "input_ids"), ("loss_mask", "loss_mask")),
        )
    )
    offline = OfflineDataProvider(
        modality="text",
        normalizer_id=normalizer,
        build_reader=partial(canonical_data.build_reader, strategy=family, keys=keys),
        build_normalizer=partial(
            canonical_data.build_normalizer,
            keys=keys,
            loss_mask_filter=requirements.loss_mask_filter,
        ),
        build_collator=partial(canonical_data.build_collator, keys=keys),
        capture_layout=layout,
    )
    streaming = ()
    if layout is not None:
        server_layout = ServerCaptureLayout(
            aux_feature=layout.aux_feature,
            last_hidden_feature=layout.last_hidden_feature,
            passthrough=(
                ("input_ids", "input_ids", ()),
                ("loss_mask", "loss_mask", ()),
            ),
        )
        streaming = (
            ServerStreamingProvider(
                modality="text",
                capture_method=requirements.capture_method,
                target_representation=target_repr,
                layout=server_layout,
                build_collator=partial(canonical_data.build_collator, keys=keys),
                select_layout=None,
            ),
        )
        contracts.append(
            replace(contracts[0], mode=FeatureMode.STREAMING, storage=None)
        )
    vocab = requirements.supports_vocab_mapping
    if cfg.model.vocab_mapping_path and not vocab:
        raise ValueError("model.vocab_mapping_path requires a draft vocabulary")
    capabilities = AlgorithmCapabilities(
        attention_backends=requirements.attention_backends,
        required_batch_size=requirements.required_batch_size,
        supports_vocab_mapping=vocab,
        supports_liger_kernel=True,
        # a checkpoint import's legacy config becomes the draft; the reader checks it
        allows_aux_layer_override=bool(init and init.checkpoint),
    )
    spec = AlgorithmSpec(
        name=family,
        draft=DraftRequirement(
            compatible_architectures=frozenset({draft.architecture}),
            default_architecture=draft.architecture,
        ),
        feature_contracts=tuple(contracts),
        capabilities=capabilities,
    )
    provider = ModelProvider(
        build_draft=None,
        build_training_model=None,
        build_bundle=build_bundle,
        needs_input_tools=online_needs_input_tools,
        default_dataloader_num_workers=None,
        loss_mask_filter=requirements.loss_mask_filter,
        draft_config=DraftConfigProvider(architecture=draft.architecture),
        resolve_capture_layers=capture_layers,
        minimum_loss_tokens=partial(
            minimum_loss_tokens, count=requirements.minimum_loss_tokens
        ),
    )
    step = StepProvider(
        build=build_step,
        options=empty_options,
        resume_contract=empty_resume_contract,
        allowed_missing_checkpoint_keys=no_missing_checkpoint_keys,
        uses_external_target_head=False,
    )
    providers = AlgorithmProviders(
        algorithm_name=family,
        model=provider,
        step=step,
        offline=(offline,),
        server_streaming=streaming,
        vocab_mapping_modes=spec.feature_modes if vocab else frozenset(),
    )
    return cfg, make_registration(spec, providers)


def build_bundle(cfg, algorithm):
    import torch

    from specforge.algorithms.common.providers import (
        MODEL_PROVENANCE_CONTRACT_KEY,
        StepRuntimeConfig,
    )
    from specforge.algorithms.objectives import build_objective_model, resolve_objective
    from specforge.checkpoint.target import TargetWeights
    from specforge.config.draft.codec import encode_config
    from specforge.config.draft.codec.target import (
        bound_vocab_size,
        resolve_target_logits,
    )
    from specforge.config.schema import ModelConfig
    from specforge.loading import initialize_draft, load_draft
    from specforge.modeling.draft.bindings import BoundLMHead
    from specforge.modeling.draft.layers.kernels import configure_kernels
    from specforge.training.assembly import ModelBundle, _device
    from specforge.training.model_loading import training_target_config

    target = training_target_config(cfg)
    draft = resolve_config(cfg, target_config=target)
    resolved = resolve_objective(draft.architecture, **_objective_settings(cfg))
    dtype, device = getattr(torch, cfg.model.torch_dtype), _device()
    init = cfg.initialization
    if (
        resolved.registration.training.explicit_initialization
        and init is None
        and not cfg.resume_path
    ):
        raise ValueError(
            f"{resolved.name} requires an explicit initialization source; "
            "set initialization: {} to train from scratch, or use resume"
        )
    source = init.checkpoint if init else None
    if cfg.resume_path:
        from specforge.training.checkpoint import CheckpointManager

        source = CheckpointManager.resolve_resume_dir(cfg.resume_path)
    target_weights = TargetWeights(
        cfg.model.target_model_path,
        cache_dir=cfg.model.cache_dir,
        keys={
            name: getattr(cfg.model, field)
            for name, field in (
                ("embed_tokens", "embedding_key"),
                ("lm_head", "lm_head_key"),
            )
            if getattr(cfg.model, field) != ModelConfig.model_fields[field].default
        },
    )
    common = dict(
        target=target_weights,
        draft=draft,
        device=device,
        dtype=dtype,
        backend=cfg.training.attention_backend,
        cache_dir=cfg.model.cache_dir,
    )
    if source:
        model = load_draft(
            source,
            **common,
            adapter=init.adapter if init else None,
            legacy_config=init.legacy_config if init else None,
            legacy_settings=_source_legacy_settings(cfg, source),
        ).train()
    else:
        weights = init.weights_from if init else None
        from_target = dict(init.from_target or {}) if init else {}
        cast_copies = ()
        # an explicit copy of the embedding, or of its tied alias, takes precedence
        explicit = ("", "embed_tokens", "embed_tokens.weight") + (
            ("lm_head", "lm_head.weight") if draft.tie_word_embeddings else ()
        )
        if (
            weights is None
            and cfg.model.load_target_embedding
            and "embed_tokens" not in draft.target_bindings
            and not any(k in from_target for k in explicit)
        ):
            from_target["embed_tokens.weight"] = target_weights.resolve(
                "embed_tokens"
            ).key
            # legacy load_embedding parity: follow model.torch_dtype
            cast_copies = ("embed_tokens.weight",)
        model = initialize_draft(
            **common,
            weights_from=weights.checkpoint if weights else None,
            mapping=weights.mapping if weights else None,
            allow_new=weights.allow_new if weights else (),
            require_loaded=tuple(cfg.training.frozen)
            + getattr(resolved.objective, "frozen_parameters", ()),
            from_target=from_target or None,
            cast_target_copies=cast_copies,
            legacy_config=init.legacy_config if init else None,
            legacy_settings=_legacy_settings(cfg),
        )
    teacher = None
    if resolved.requires_teacher:
        transform = resolve_target_logits(target_weights.config)
        if isinstance(model.lm_head, BoundLMHead) and model.lm_head.config == transform:
            teacher = model.lm_head
        else:
            teacher = BoundLMHead(
                transform,
                vocab_size=bound_vocab_size(target),
                hidden_size=target["hidden_size"],
                device=device,
                dtype=dtype,
            )
            teacher.load_target_weight(target_weights.resolve("lm_head").read())
    if teacher is not None and hasattr(model, "vocabulary"):
        width = bound_vocab_size(target)
        if model.vocabulary.vocab_size != width:
            # t2d must cover every teacher logit column
            raise ValueError(
                f"draft vocab_size {model.vocabulary.vocab_size} must equal the "
                f"target logit width {width} (padded_vocab_size when the target pads)"
            )
    use_liger = cfg.model.use_liger_kernel is True
    configure_kernels(
        model, use_liger=use_liger, use_fused_conv=cfg.model.use_fused_conv
    )
    if hasattr(model, "vocabulary") and cfg.model.vocab_mapping_path:
        mapping = torch.load(
            cfg.model.vocab_mapping_path, map_location=device, weights_only=True
        )
        # a loaded or copied vocabulary was trained with its own mapping
        if bool(model.vocabulary.ready) and (
            not torch.equal(model.vocabulary.t2d, mapping["t2d"])
            or not torch.equal(model.vocabulary.d2t, mapping["d2t"])
        ):
            raise ValueError(
                "vocab_mapping_path disagrees with the loaded draft vocabulary"
            )
        model.vocabulary.load_mapping(t2d=mapping["t2d"], d2t=mapping["d2t"])
    runtime = build_objective_model(
        model,
        **_objective_settings(cfg),
        frozen=cfg.training.frozen,
        target_head=teacher,
    )
    runtime.frozen_target_modules = tuple(
        dict.fromkeys(
            [model.get_submodule(name) for name in draft.target_bindings]
            + ([teacher] if teacher is not None else [])
        )
    )
    runtime.fsdp_param_dtype = dtype
    contract = {
        "canonical_draft_config": encode_config(draft),
        "canonical_objective": runtime.objective_name,
        "canonical_objective_options": asdict(runtime.objective),
        "canonical_use_liger_kernel": use_liger,
        "canonical_use_fused_conv": cfg.model.use_fused_conv,
        "canonical_frozen": tuple(sorted(set(cfg.training.frozen))),
        "canonical_loss": asdict(runtime.token_loss),
        "canonical_weighting": asdict(runtime.weighting),
        "canonical_loss_execution": asdict(runtime.execution),
        "canonical_loss_implementations": runtime.loss_implementations.names,
    }
    from specforge.training.provenance import model_resume_provenance

    provenance = model_resume_provenance(
        cfg, draft, target, capture_layers=capture_layers(cfg, draft, target)
    )
    contract[MODEL_PROVENANCE_CONTRACT_KEY] = tuple(sorted(provenance.items()))
    return ModelBundle(
        model=runtime,
        draft_model=model,
        draft_config=draft,
        target_head=teacher,
        target_hidden_size=target["hidden_size"],
        target_vocab_size=bound_vocab_size(target),
        draft_vocab_size=draft.draft_vocab_size,
        capture_layers=capture_layers(cfg, draft, target),
        strategy_kwargs=StepRuntimeConfig(
            options={
                "name": algorithm.name,
                "objective": runtime.objective_name,
                "required_features": tuple(runtime.registration.inputs),
            },
            resume_contract=contract,
            allowed_missing_checkpoint_keys=frozenset(),
        ),
    )
