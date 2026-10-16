"""Compose KV context inputs with an existing DFlash-family objective."""

from dataclasses import replace
from functools import partial

from specforge.algorithms.common import target_kv_data as data
from specforge.algorithms.common.providers import (
    make_registration,
    resolve_server_capture_layout,
)
from specforge.algorithms.contracts import FeatureMode


def _build_step(original, wrapped_model, **options):
    step = original(wrapped_model, **options)
    step.configure_conditioning("kv")
    return step


def _resume_contract(original, config, draft_model, training_model):
    from specforge.modeling.draft.target_kv import target_kv_config

    contract = original(config, draft_model, training_model)
    contract.update(
        conditioning_source="kv",
        target_kv_state="post_qk_norm_post_rope",
        target_kv_geometry=target_kv_config(draft_model.config),
    )
    return contract


def _build_training_model(
    original, config, draft_model, draft_config, target_config, tokenizer
):
    from specforge.modeling.draft.target_kv import validate_target_kv_config

    validate_target_kv_config(draft_config, target_config)
    for path in (config.data.hidden_states_path, config.data.eval_hidden_states_path):
        if path:
            data.validate_manifest(
                path, draft_config, target_config, config.model.target_model_path
            )
    return original(config, draft_model, draft_config, target_config, tokenizer)


def _capture_options(draft_provider, config):
    from specforge.modeling.draft.target_kv import capture_geometry
    from specforge.modeling.target.target_utils import load_target_config
    from specforge.training.model_loading import draft_config_dict

    target = load_target_config(
        config.model.target_model_path,
        cache_dir=config.model.cache_dir,
        trust_remote_code=config.model.trust_remote_code,
    )
    if getattr(target, "model_type", None) != "qwen3":
        raise ValueError("online target-KV capture currently supports Qwen3 only")
    draft = draft_config_dict(config, provider=draft_provider)
    return {"target_kv_geometry": capture_geometry(draft, target)}


def configure_kv_conditioning(config, algorithm):
    """Return a new registration; preserve the immutable hidden-input catalog."""
    if config.model.input_modality != "text":
        raise ValueError("kv conditioning currently supports text only")
    base = algorithm.providers
    offline = base.offline_for("text")
    streaming = base.server_streaming_for("text")
    layout = resolve_server_capture_layout(algorithm, config, modality="text")
    include_teacher = layout.last_hidden_feature is not None
    required = set(data.feature_names(include_teacher=include_teacher))
    collator = partial(data.build_collator, include_teacher=include_teacher)
    contracts = []
    for contract in algorithm.spec.feature_contracts:
        storage = None
        if contract.mode == FeatureMode.OFFLINE:
            storage = replace(
                contract.storage,
                format=data.CAPTURE_FORMAT,
                required_tensors=required,
            )
        contracts.append(replace(contract, required_tensors=required, storage=storage))

    # Retain the algorithm's provider identity; conditioning only changes its
    # feature format and executable adapters. No normalizer-ID dispatch is used.
    offline = replace(
        offline,
        build_reader=partial(
            data.build_reader,
            strategy=algorithm.name,
            include_teacher=include_teacher,
            target_repr=streaming.target_representation,
        ),
        build_normalizer=partial(
            data.build_normalizer, include_teacher=include_teacher
        ),
        build_collator=collator,
        capture_layout=None,  # KV uses the dedicated capture preparation script.
    )
    streaming = replace(
        streaming,
        # Existing versioned SGLang extension uses this wire protocol name.
        capture_method="dspark_kv",
        layout=replace(layout, aux_feature="target_kv"),
        select_layout=None,  # Teacher selection is already resolved above.
        build_collator=collator,
        capture_options=partial(_capture_options, base.model.draft_config),
    )
    providers = replace(
        base,
        step=replace(
            base.step,
            build=partial(_build_step, base.step.build),
            resume_contract=partial(_resume_contract, base.step.resume_contract),
        ),
        model=replace(
            base.model,
            build_training_model=partial(
                _build_training_model, base.model.build_training_model
            ),
        ),
        offline=(offline,),
        server_streaming=(streaming,),
    )
    return make_registration(
        replace(algorithm.spec, feature_contracts=tuple(contracts)), providers
    )
