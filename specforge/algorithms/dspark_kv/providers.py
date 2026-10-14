"""Captured target-KV conditioning with the existing DSpark objectives."""

from dataclasses import replace

from specforge.algorithms.common.providers import (
    OfflineDataProvider,
    ServerCaptureLayout,
    ServerStreamingProvider,
    make_registration,
)
from specforge.algorithms.contracts import (
    FeatureContract,
    FeatureMode,
    OfflineStorageContract,
)
from specforge.algorithms.dspark import providers as dspark
from specforge.algorithms.dspark_kv import data


def build_step(wrapped_model, **_options):
    from specforge.training.strategies.base import DSparkKVTrainStrategy

    return DSparkKVTrainStrategy(wrapped_model)


def apply_draft_overrides(_config, draft_config):
    from specforge.modeling.draft.target_kv import target_kv_config

    if target_kv_config(draft_config) is None:
        raise ValueError("dspark_kv requires an explicit target_kv draft config")


def resolve_capture_layers(config, draft_config, target_config):
    from specforge.modeling.draft.target_kv import validate_target_kv_config

    validate_target_kv_config(draft_config, target_config)
    return dspark.resolve_capture_layers(config, draft_config, target_config)


def capture_options(config):
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
    draft = draft_config_dict(
        config, provider=dspark.algorithm_providers().model.draft_config
    )
    return {"target_kv_geometry": capture_geometry(draft, target)}


def build_training_model(config, draft_model, draft_config, target_config, tokenizer):
    from specforge.modeling.draft.target_kv import validate_target_kv_config

    validate_target_kv_config(draft_config, target_config)
    for path in (config.data.hidden_states_path, config.data.eval_hidden_states_path):
        if path:
            data.validate_manifest(
                path, draft_config, target_config, config.model.target_model_path
            )
    return dspark.build_training_model(
        config, draft_model, draft_config, target_config, tokenizer
    )


def resume_contract(config, draft_model, training_model):
    from specforge.modeling.draft.target_kv import target_kv_config

    contract = {
        name.replace("dspark_", "dspark_kv_", 1): value
        for name, value in dspark.resume_contract(
            config, draft_model, training_model
        ).items()
    }
    contract.update(
        dspark_kv_conditioning_source="target_kv",
        dspark_kv_state="post_qk_norm_post_rope",
        dspark_kv_geometry=target_kv_config(draft_model.config),
    )
    return contract


def create_registration():
    base_spec = dspark.algorithm_spec()
    spec = replace(
        base_spec,
        name="dspark_kv",
        feature_contracts=(
            FeatureContract(
                mode=FeatureMode.STREAMING,
                modality="text",
                required_tensors=set(data.FEATURES),
                allowed_target_representations={"hidden_state"},
                default_target_representation="hidden_state",
            ),
            FeatureContract(
                mode=FeatureMode.OFFLINE,
                modality="text",
                required_tensors=set(data.FEATURES),
                allowed_target_representations={"hidden_state"},
                default_target_representation="hidden_state",
                storage=OfflineStorageContract(
                    format=data.NORMALIZER_ID,
                    required_tensors=set(data.FEATURES),
                    normalizer=data.NORMALIZER_ID,
                ),
            ),
        ),
    )
    base = dspark.algorithm_providers()
    providers = replace(
        base,
        algorithm_name="dspark_kv",
        step=replace(base.step, build=build_step, resume_contract=resume_contract),
        model=replace(
            base.model,
            build_training_model=build_training_model,
            resolve_capture_layers=resolve_capture_layers,
            draft_config=replace(
                base.model.draft_config, apply_overrides=apply_draft_overrides
            ),
        ),
        offline=(
            OfflineDataProvider(
                modality="text",
                normalizer_id=data.NORMALIZER_ID,
                build_reader=data.build_reader,
                build_normalizer=data.build_normalizer,
                build_collator=data.build_collator,
            ),
        ),
        server_streaming=(
            ServerStreamingProvider(
                modality="text",
                capture_method="dspark_kv",
                target_representation="hidden_state",
                layout=ServerCaptureLayout(
                    aux_feature="target_kv",
                    last_hidden_feature="target_last_hidden_states",
                    passthrough=(
                        ("input_ids", "input_ids", ()),
                        ("loss_mask", "loss_mask", ()),
                    ),
                ),
                build_collator=data.build_collator,
                capture_options=capture_options,
            ),
        ),
    )
    return make_registration(spec, providers)
