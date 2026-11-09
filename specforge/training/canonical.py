"""Compose canonical models/objectives with existing data and trainer ports."""

from dataclasses import replace
from functools import partial

from .model_loading import capture_layers
from .model_loading import resolve_draft_config as resolve_config


def minimum_loss_tokens(cfg, draft):
    return 2


def build_step(wrapped_model, *, target_head=None, **options):
    from specforge.training.strategies.canonical import CanonicalTrainStrategy

    return CanonicalTrainStrategy(wrapped_model, **options)


def resolve_run(cfg, registry=None, *, target_config=None):
    """Reuse deployment/data ports while resolving structure and loss separately."""
    from specforge.algorithms.common.defaults import (
        empty_options,
        empty_resume_contract,
        no_missing_checkpoint_keys,
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
    from specforge.algorithms.objectives import (
        default_objective,
        objective_registrations,
    )
    from specforge.config.draft.codec import encode_config

    from . import canonical_data

    draft = resolve_config(cfg, target_config=target_config)
    default = default_objective(draft.architecture)
    family = "dflash" if default in ("dflash2", "dmtp") else default
    objective = cfg.training.objective or default
    entry = objective_registrations().get(objective)
    if entry is None or draft.architecture not in entry.architectures:
        raise ValueError(
            f"objective {objective!r} is incompatible with {draft.architecture}"
        )
    objective_options = entry.factory(**cfg.training.objective_options)
    backend = cfg.training.attention_backend
    if cfg.training.tp_size != 1:
        raise ValueError("canonical training does not implement tensor parallelism")
    sp_size = cfg.training.sp_ulysses_size * cfg.training.sp_ring_size
    if sp_size > 1 and (default != "eagle3" or backend != "usp"):
        raise ValueError("sequence parallelism requires EAGLE3 with the USP backend")
    if backend in ("fa", "usp"):
        from specforge.config.draft.attention import GQASpec, MHASpec

        if default != "eagle3" or any(
            type(a) not in (GQASpec, MHASpec)
            or not a.is_causal
            or a.sliding_window is not None
            or a.attention_dropout != 0
            for a in draft.attention_configs.values()
        ):
            raise ValueError(
                "FA/USP requires EAGLE3 with causal, full GQA/MHA attention"
            )
        if backend == "usp" and any(
            a.num_attention_heads % cfg.training.sp_ulysses_size
            or a.num_key_value_heads % cfg.training.sp_ulysses_size
            for a in draft.attention_configs.values()
        ):
            raise ValueError("USP head counts must be divisible by the Ulysses degree")
    if getattr(objective_options, "compact_teacher", False) and cfg.mode != "offline":
        raise ValueError("compact_teacher currently requires offline features")
    # The capture family remains stable in data/rendezvous identifiers.
    training = cfg.training.model_copy(update={"objective": objective})
    cfg = cfg.model_copy(update={"draft": encode_config(draft), "training": training})
    keys = tuple(entry.inputs)
    if default in ("dflash", "dflash2") and objective_options.teacher_metrics:
        keys += ("target_last_hidden_states",)
    # Capture protocol names describe target outputs, independently of the
    # draft constructor and the selected objective.
    capture_method = (
        "eagle3"
        if default in ("eagle3", "peagle")
        else "dspark" if default in ("dspark", "dspine") else "dflash"
    )
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
            capture_method=capture_method,
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
                capture_method=capture_method,
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
        attention_backends=frozenset(
            {"eager", "sdpa", "flex_attention", "fa", "usp"}
            if default == "eagle3"
            else {"eager", "sdpa", "flex_attention"}
        ),
        supports_compact_teacher=default == "eagle3",
        supports_trim_loss_positions=default == "eagle3",
        supports_liger_kernel=default
        in ("dflash", "dflash2", "domino", "dspark", "dspine", "dmtp"),
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
    from specforge.data.loss_mask import has_consecutive_supervised_tokens

    provider = ModelProvider(
        draft_config=DraftConfigProvider(architecture=draft.architecture),
        resolve_capture_layers=capture_layers,
        minimum_loss_tokens=minimum_loss_tokens,
        default_dataloader_num_workers=4 if vocab else 8,
        loss_mask_filter=None if vocab else has_consecutive_supervised_tokens,
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
    return cfg, (
        registry.resolve(family)
        if registry is not None
        else make_registration(spec, providers)
    )
