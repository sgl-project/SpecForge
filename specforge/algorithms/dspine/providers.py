"""DSpine uses the existing target-feature capture and training lifecycle."""

from dataclasses import asdict
from functools import partial

from specforge.algorithms.common.defaults import (
    empty_options,
    no_missing_checkpoint_keys,
)
from specforge.algorithms.common.hidden_states_data import (
    DSPARK_NORMALIZER_ID,
    build_dspark_collator,
    build_dspark_offline_normalizer,
    build_dspark_offline_reader,
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
from specforge.data.loss_mask import has_consecutive_supervised_tokens


def build_step(wrapped_model, **_options):
    from .strategy import DSpineTrainStrategy

    return DSpineTrainStrategy(wrapped_model)


def build_draft(config, draft_config):
    from specforge.algorithms.model_providers import build_registered_draft

    if config.model.use_liger_kernel:
        from specforge.algorithms.dflash.providers import resolve_dflash_kernels
        from specforge.algorithms.model_providers import build_dflash_draft

        return build_dflash_draft(
            config, draft_config, resolve_dflash_kernels(config, draft_config)
        )
    return build_registered_draft(config, draft_config)


def build_training_model(config, draft_model, draft_config, target_config, tokenizer):
    from specforge.algorithms.model_providers import build_dspine_model

    return build_dspine_model(
        config,
        draft_model,
        draft_config,
        target_config,
        tokenizer,
    )


def resume_contract(_config, draft_model, training_model):
    return {
        "dspine_attention_layout": (
            tuple(draft_model.layer_types),
            draft_model.sliding_window,
            True,
        ),
        "dspine_options": tuple(sorted(asdict(draft_model.dspine_config).items())),
        "dspine_block_size": draft_model.block_size,
        "dspine_target_layer_ids": tuple(draft_model.target_layer_ids),
        "dspine_num_hidden_layers": len(draft_model.layers),
        "dspine_num_anchors": training_model.num_anchors,
        "dspine_mask_token_id": training_model.mask_token_id,
        "dspine_attention_backend": training_model.attention_backend,
        "dspine_loss_decay_gamma": training_model.loss_decay_gamma,
    }


def resolve_capture_layers(config, draft_config, target_config):
    from specforge.algorithms.model_providers import resolve_dflash_capture_layers

    return resolve_dflash_capture_layers(config, draft_config, target_config)


def minimum_loss_tokens(config, draft_config):
    from specforge.algorithms.model_providers import dflash_min_loss_tokens

    return dflash_min_loss_tokens(config, draft_config)


def needs_input_tools(config, draft_model):
    from specforge.algorithms.model_providers import dflash_needs_input_tools

    return dflash_needs_input_tools(config, draft_model)


def algorithm_spec() -> AlgorithmSpec:
    ready = {
        "input_ids",
        "loss_mask",
        "hidden_states",
        "target_last_hidden_states",
    }
    return AlgorithmSpec(
        name="dspine",
        draft=DraftRequirement(
            compatible_architectures=frozenset({"DSpineDraftModel"}),
            default_architecture="DSpineDraftModel",
        ),
        feature_contracts=(
            FeatureContract(
                mode=FeatureMode.OFFLINE,
                modality="text",
                required_tensors=ready,
                allowed_target_representations={"hidden_state"},
                default_target_representation="hidden_state",
                storage=OfflineStorageContract(
                    format="specforge_hidden_states_v1",
                    required_tensors=ready,
                    normalizer=DSPARK_NORMALIZER_ID,
                ),
            ),
            FeatureContract(
                mode=FeatureMode.STREAMING,
                modality="text",
                required_tensors=ready,
                allowed_target_representations={"hidden_state"},
                default_target_representation="hidden_state",
            ),
        ),
        capabilities=AlgorithmCapabilities(
            attention_backends={"eager", "sdpa", "flex_attention"},
            supports_liger_kernel=True,
        ),
    )


def algorithm_providers() -> AlgorithmProviders:
    return AlgorithmProviders(
        algorithm_name="dspine",
        step=StepProvider(
            build=build_step,
            options=empty_options,
            resume_contract=resume_contract,
            allowed_missing_checkpoint_keys=no_missing_checkpoint_keys,
            uses_external_target_head=False,
        ),
        model=ModelProvider(
            draft_config=DraftConfigProvider(
                architecture="DSpineDraftModel",
                compatible_architectures=frozenset({"DSpineDraftModel"}),
                expected_auto_map_model="dspine.DSpineDraftModel",
            ),
            build_draft=build_draft,
            build_training_model=build_training_model,
            resolve_capture_layers=resolve_capture_layers,
            minimum_loss_tokens=minimum_loss_tokens,
            needs_input_tools=needs_input_tools,
            default_dataloader_num_workers=8,
            loss_mask_filter=has_consecutive_supervised_tokens,
        ),
        offline=(
            OfflineDataProvider(
                modality="text",
                normalizer_id=DSPARK_NORMALIZER_ID,
                capture_layout=OfflineCaptureLayout(
                    capture_method="dspark",
                    aux_feature="hidden_states",
                    last_hidden_feature="target_last_hidden_states",
                    passthrough=(
                        ("input_ids", "input_ids"),
                        ("loss_mask", "loss_mask"),
                    ),
                ),
                build_reader=partial(build_dspark_offline_reader, "dspine"),
                build_normalizer=build_dspark_offline_normalizer,
                build_collator=build_dspark_collator,
            ),
        ),
        server_streaming=(
            ServerStreamingProvider(
                modality="text",
                capture_method="dspark",
                target_representation="hidden_state",
                layout=ServerCaptureLayout(
                    aux_feature="hidden_states",
                    last_hidden_feature="target_last_hidden_states",
                    passthrough=(
                        ("input_ids", "input_ids", ()),
                        ("loss_mask", "loss_mask", ()),
                    ),
                ),
                build_collator=build_dspark_collator,
            ),
        ),
    )


def create_registration():
    return make_registration(algorithm_spec(), algorithm_providers())
