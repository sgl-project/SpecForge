"""Compose canonical models/objectives with existing data and trainer ports."""

from dataclasses import asdict, replace
from functools import partial
from pathlib import Path


def enabled(cfg):
    return (
        cfg.draft is not None
        or cfg.initialization is not None
        or cfg.resume is not None
    )


def resolve_config(cfg, *, target_config=None):
    from specforge.config.draft.codec import parse_draft

    draft = cfg.draft
    if isinstance(draft, str):
        import yaml

        draft = yaml.safe_load(Path(draft).read_text())
    supplied = (
        parse_draft(draft, target_config=target_config) if draft is not None else None
    )
    initialization = cfg.initialization
    source = initialization.checkpoint if initialization else None
    if cfg.training.resume_from:
        from specforge.training.checkpoint import CheckpointManager

        source = CheckpointManager.resolve_resume_dir(cfg.training.resume_from)
    if source:
        from specforge.loading import read_draft

        loaded = read_draft(
            source,
            target=cfg.model.target_model_path,
            cache_dir=cfg.model.cache_dir,
            adapter=initialization.adapter if initialization else None,
            legacy_config=initialization.legacy_config if initialization else None,
        ).config
        if supplied is not None and supplied != loaded:
            raise ValueError(
                "draft must exactly match checkpoint structure; use weights_from to change it"
            )
        return loaded
    if supplied is None:
        raise ValueError("canonical training requires draft or a checkpoint")
    return supplied


def capture_layers(cfg, draft, target_config):
    from specforge.config.draft.codec import parse_draft
    from specforge.modeling.target.target_utils import target_text_config

    if isinstance(draft, dict):
        draft = parse_draft(
            draft,
            target_config=target_text_config(target_config).to_dict(),
        )
    features = draft.spec_config.target_features
    if hasattr(features, "target_layer_ids"):
        return list(features.target_layer_ids)
    target = getattr(target_config, "text_config", target_config)
    return [target.num_hidden_layers - 1]


def minimum_loss_tokens(cfg, draft):
    return 2


def build_step(wrapped_model, *, target_head=None, **options):
    from specforge.training.strategies.canonical import CanonicalTrainStrategy

    return CanonicalTrainStrategy(wrapped_model, **options)


def _objective_settings(cfg):
    return dict(
        name=cfg.training.objective,
        options=cfg.training.objective_options,
        loss=cfg.training.loss,
        weighting=cfg.training.weighting,
    )


def resolve_run(cfg, registry, *, target_config=None):
    """Reuse deployment/data ports while resolving structure and loss separately."""
    from specforge.algorithms.common.defaults import (
        empty_options,
        empty_resume_contract,
        no_missing_checkpoint_keys,
    )
    from specforge.algorithms.common.providers import (
        AlgorithmProviders,
        DraftConfigProvider,
        OfflineCaptureLayout,
        OfflineDataProvider,
        ServerCaptureLayout,
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
    from specforge.algorithms.objectives import default_objective, resolve_objective
    from specforge.config.draft.codec import encode_config

    from . import canonical_data

    draft = resolve_config(cfg, target_config=target_config)
    default = default_objective(draft.architecture)
    family = "dflash" if default in ("dflash2", "dmtp") else default
    if cfg.training.strategy not in ("eagle3", family, default):
        raise ValueError("training.strategy conflicts with the draft architecture")
    # New objectives have their own options. Never silently consume legacy
    # algorithm switches with a different meaning on the canonical path.
    legacy_options = {
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
        "norm_before_residual",
    }
    from specforge.config.schema import TrainingConfig

    defaults = TrainingConfig()
    present = {
        name
        for name in legacy_options
        if getattr(cfg.training, name, None) != getattr(defaults, name, None)
    }
    if present:
        raise ValueError(
            f"canonical training uses training.loss, training.weighting and "
            f"training.objective_options, not {sorted(present)}"
        )
    resolved = resolve_objective(draft.architecture, **_objective_settings(cfg))
    objective, entry = resolved.name, resolved.registration
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
    base = registry.resolve(family)
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
        if default == "dmtp"
        else OfflineCaptureLayout(
            capture_method=base.providers.server_streaming_for("text").capture_method,
            aux_feature="hidden_states" if "hidden_states" in keys else None,
            last_hidden_feature="target_last_hidden_states" if target_repr else None,
            passthrough=(("input_ids", "input_ids"), ("loss_mask", "loss_mask")),
        )
    )
    offline = OfflineDataProvider(
        modality="text",
        normalizer_id=normalizer,
        build_reader=partial(canonical_data.build_reader, strategy=family, keys=keys),
        build_normalizer=partial(canonical_data.build_normalizer, keys=keys),
        build_collator=partial(canonical_data.build_collator, keys=keys),
        capture_layout=layout,
    )
    streaming = ()
    if default != "dmtp":
        original = base.providers.server_streaming_for("text")
        server_layout = ServerCaptureLayout(
            aux_feature=layout.aux_feature,
            last_hidden_feature=layout.last_hidden_feature,
            passthrough=(
                ("input_ids", "input_ids", ()),
                ("loss_mask", "loss_mask", ()),
            ),
        )
        streaming = (
            replace(
                original,
                target_representation=target_repr,
                layout=server_layout,
                build_collator=partial(canonical_data.build_collator, keys=keys),
                select_layout=None,
            ),
        )
        contracts.append(
            replace(contracts[0], mode=FeatureMode.STREAMING, storage=None)
        )
    vocab = default in ("eagle3", "peagle")
    capabilities = AlgorithmCapabilities(
        attention_backends=frozenset({"eager"}),
        required_batch_size=1 if default == "peagle" else None,
        supports_vocab_mapping=vocab,
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
    provider = replace(
        base.providers.model,
        draft_config=DraftConfigProvider(architecture=draft.architecture),
        resolve_capture_layers=capture_layers,
        minimum_loss_tokens=minimum_loss_tokens,
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
    from specforge.config.draft.codec import encode_config
    from specforge.config.draft.codec.target import resolve_target_logits
    from specforge.loading import initialize_draft, load_draft
    from specforge.modeling.draft.bindings import BoundLMHead
    from specforge.modeling.target.target_head import TargetHead
    from specforge.modeling.target.target_utils import (
        load_target_config,
        target_hidden_size,
        target_text_config,
        target_vocab_size,
    )
    from specforge.training.assembly import ModelBundle, _device

    target = load_target_config(
        cfg.model.target_model_path,
        cache_dir=cfg.model.cache_dir,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    draft = resolve_config(cfg, target_config=target_text_config(target).to_dict())
    resolved = resolve_objective(draft.architecture, **_objective_settings(cfg))
    dtype, device = getattr(torch, cfg.model.torch_dtype), _device()
    init = cfg.initialization
    source = init.checkpoint if init else None
    if cfg.training.resume_from:
        from specforge.training.checkpoint import CheckpointManager

        source = CheckpointManager.resolve_resume_dir(cfg.training.resume_from)
    common = dict(
        target=cfg.model.target_model_path,
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
        ).train()
    else:
        weights = init.weights_from if init else None
        model = initialize_draft(
            **common,
            weights_from=weights.checkpoint if weights else None,
            mapping=weights.mapping if weights else None,
            allow_new=weights.allow_new if weights else (),
            from_target=init.from_target if init else None,
            legacy_config=init.legacy_config if init else None,
        )
    teacher = None
    if resolved.requires_teacher:
        loaded = TargetHead.from_pretrained(
            cfg.model.target_model_path,
            lm_head_key=cfg.model.lm_head_key,
            embedding_key=cfg.model.embedding_key,
            dtype=dtype,
            cache_dir=cfg.model.cache_dir,
            trust_remote_code=cfg.model.trust_remote_code,
        )
        teacher = BoundLMHead(
            resolve_target_logits(target.to_dict()),
            vocab_size=target_vocab_size(target),
            hidden_size=target_hidden_size(target),
            device=device,
            dtype=dtype,
        )
        teacher.load_target_weight(loaded.fc.weight)
    if hasattr(model, "vocabulary") and cfg.model.vocab_mapping_path:
        mapping = torch.load(
            cfg.model.vocab_mapping_path, map_location=device, weights_only=True
        )
        if (
            source
            and bool(model.vocabulary.ready)
            and (
                not torch.equal(model.vocabulary.t2d, mapping["t2d"])
                or not torch.equal(model.vocabulary.d2t, mapping["d2t"])
            )
        ):
            raise ValueError("vocab_mapping_path disagrees with checkpoint vocabulary")
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
        "canonical_frozen": tuple(cfg.training.frozen),
        "canonical_loss": asdict(runtime.token_loss),
        "canonical_weighting": asdict(runtime.weighting),
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
        target_hidden_size=target_hidden_size(target),
        target_vocab_size=target_vocab_size(target),
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
