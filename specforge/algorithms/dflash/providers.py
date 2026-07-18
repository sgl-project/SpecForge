"""Built-in DFlash registration and executable providers."""

from __future__ import annotations

from functools import partial

from specforge.algorithms.common.defaults import (
    empty_options,
    no_missing_checkpoint_keys,
)
from specforge.algorithms.common.dflash_family_data import (
    NORMALIZER_ID,
    build_collator,
    build_offline_normalizer,
    build_offline_reader,
)
from specforge.algorithms.common.providers import (
    AlgorithmProviders,
    DraftConfigProvider,
    ModelProvider,
    OfflineDataProvider,
    ServerCaptureLayout,
    ServerStreamingProvider,
    StepProvider,
    TargetDerivedDraftDefaults,
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

ALGORITHM_NAME = "dflash"
DRAFT_ARCHITECTURE = "DFlashDraftModel"


def build_step(wrapped_model, *, target_head=None, **_options):
    del target_head
    from specforge.training.strategies.base import DFlashTrainStrategy

    return DFlashTrainStrategy(wrapped_model)


def resume_contract(_config, draft_model, training_model):
    """Persist resolved DFlash architecture, sampling, and loss semantics."""

    return {
        "dflash_draft_num_hidden_layers": int(draft_model.config.num_hidden_layers),
        "dflash_target_layer_ids": tuple(
            int(layer_id) for layer_id in draft_model.target_layer_ids
        ),
        "dflash_block_size": int(training_model.block_size),
        "dflash_mask_token_id": int(training_model.mask_token_id),
        "dflash_attention_backend": str(training_model.attention_backend),
        "dflash_num_anchors": int(training_model.num_anchors),
        "dflash_loss_decay_gamma": training_model.loss_decay_gamma,
        "dflash_loss_type": str(training_model.loss_type),
        "dflash_dpace_alpha": float(training_model.dpace_alpha),
    }


def build_draft(config, draft_config):
    from specforge.algorithms.model_providers import build_registered_draft

    return build_registered_draft(config, draft_config)


def build_training_model(config, draft_model, draft_config, target_config, tokenizer):
    from specforge.algorithms.model_providers import build_dflash_model

    return build_dflash_model(
        config,
        draft_model,
        draft_config,
        target_config,
        tokenizer,
    )


def resolve_capture_layers(config, draft_config, target_config):
    from specforge.algorithms.model_providers import resolve_dflash_capture_layers

    return resolve_dflash_capture_layers(config, draft_config, target_config)


def populate_target_defaults(payload, target_config, config):
    from specforge.algorithms.model_providers import populate_dflash_generated_config

    return populate_dflash_generated_config(payload, target_config, config)


def apply_draft_overrides(config, draft_config):
    from specforge.algorithms.model_providers import apply_dflash_overrides

    return apply_dflash_overrides(config, draft_config)


def minimum_loss_tokens(config, draft_config):
    from specforge.algorithms.model_providers import dflash_min_loss_tokens

    return dflash_min_loss_tokens(config, draft_config)


def needs_input_tools(config, draft_model):
    from specforge.algorithms.model_providers import dflash_needs_input_tools

    return dflash_needs_input_tools(config, draft_model)


def algorithm_spec() -> AlgorithmSpec:
    ready = {"input_ids", "loss_mask", "hidden_states"}
    return AlgorithmSpec(
        name=ALGORITHM_NAME,
        draft=DraftRequirement(
            compatible_architectures={DRAFT_ARCHITECTURE},
            default_architecture=DRAFT_ARCHITECTURE,
            supported_overrides={"num_hidden_layers", "block_size"},
        ),
        feature_contracts=(
            FeatureContract(
                mode=FeatureMode.OFFLINE,
                modality="text",
                required_tensors=ready,
                storage=OfflineStorageContract(
                    format="specforge_hidden_states_v1",
                    required_tensors=ready,
                    normalizer=NORMALIZER_ID,
                ),
            ),
            FeatureContract(
                mode=FeatureMode.STREAMING,
                modality="text",
                required_tensors=ready,
            ),
        ),
        capabilities=AlgorithmCapabilities(
            attention_backends={"eager", "sdpa", "flex_attention"},
        ),
    )


def algorithm_providers() -> AlgorithmProviders:
    collator = build_collator
    return AlgorithmProviders(
        algorithm_name=ALGORITHM_NAME,
        step=StepProvider(
            build=build_step,
            options=empty_options,
            resume_contract=resume_contract,
            allowed_missing_checkpoint_keys=no_missing_checkpoint_keys,
            uses_external_target_head=False,
        ),
        model=ModelProvider(
            draft_config=DraftConfigProvider(
                architecture=DRAFT_ARCHITECTURE,
                expected_auto_map_model="dflash.DFlashDraftModel",
                target_defaults=TargetDerivedDraftDefaults(
                    model_type="qwen3",
                    num_hidden_layers=1,
                    populate=populate_target_defaults,
                ),
                apply_overrides=apply_draft_overrides,
            ),
            build_draft=build_draft,
            build_training_model=build_training_model,
            resolve_capture_layers=resolve_capture_layers,
            minimum_loss_tokens=minimum_loss_tokens,
            needs_input_tools=needs_input_tools,
            default_dataloader_num_workers=8,
        ),
        offline=(
            OfflineDataProvider(
                modality="text",
                normalizer_id=NORMALIZER_ID,
                build_reader=partial(build_offline_reader, ALGORITHM_NAME),
                build_normalizer=build_offline_normalizer,
                build_collator=collator,
            ),
        ),
        server_streaming=(
            ServerStreamingProvider(
                modality="text",
                capture_method="dflash",
                target_representation=None,
                layout=ServerCaptureLayout(
                    aux_feature="hidden_states",
                    last_hidden_feature=None,
                    passthrough=(
                        ("input_ids", "input_ids", ()),
                        ("loss_mask", "loss_mask", ()),
                    ),
                ),
                build_collator=collator,
            ),
        ),
    )


def create_registration():
    return make_registration(algorithm_spec(), algorithm_providers())


__all__ = ["algorithm_providers", "algorithm_spec", "create_registration"]
