"""MTP capture metadata; models and objectives use the canonical training entry."""

from __future__ import annotations

from functools import partial

from specforge.algorithms.common.defaults import (
    empty_options,
    empty_resume_contract,
    no_missing_checkpoint_keys,
    one_loss_token,
    online_needs_input_tools,
)
from specforge.algorithms.common.hidden_states_data import (
    MTP_NORMALIZER_ID,
    build_mtp_collator,
    build_mtp_offline_normalizer,
    build_mtp_offline_reader,
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

ALGORITHM_NAME = "mtp"
DRAFT_ARCHITECTURE = "MTPDraftModel"


def build_step(wrapped_model, *, target_head=None, **options):
    from specforge.training.canonical import build_step as build

    return build(wrapped_model, target_head=target_head, **options)


def canonical_assembly_required(*args, **kwargs):
    raise ValueError(
        "MTP uses the canonical training entry: supply draft and initialization "
        "(or resume); see config/draft/presets/README.md"
    )


def resolve_capture_layers(config, draft_config, target_config):
    from specforge.training.canonical import capture_layers

    return capture_layers(config, draft_config, target_config)


def create_registration():
    return make_registration(algorithm_spec(), algorithm_providers())


def algorithm_spec() -> AlgorithmSpec:
    ready = {
        "input_ids",
        "loss_mask",
        "target_last_hidden_states",
    }
    return AlgorithmSpec(
        name=ALGORITHM_NAME,
        draft=DraftRequirement(
            compatible_architectures={DRAFT_ARCHITECTURE},
            default_architecture=DRAFT_ARCHITECTURE,
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
                    normalizer=MTP_NORMALIZER_ID,
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
            attention_backends={"eager", "sdpa"},
        ),
    )


def algorithm_providers() -> AlgorithmProviders:
    return AlgorithmProviders(
        algorithm_name=ALGORITHM_NAME,
        step=StepProvider(
            build=build_step,
            options=empty_options,
            resume_contract=empty_resume_contract,
            allowed_missing_checkpoint_keys=no_missing_checkpoint_keys,
            uses_external_target_head=False,
        ),
        model=ModelProvider(
            draft_config=DraftConfigProvider(
                architecture=DRAFT_ARCHITECTURE,
            ),
            build_draft=canonical_assembly_required,
            build_training_model=canonical_assembly_required,
            resolve_capture_layers=resolve_capture_layers,
            minimum_loss_tokens=one_loss_token,
            needs_input_tools=online_needs_input_tools,
            default_dataloader_num_workers=8,
        ),
        offline=(
            OfflineDataProvider(
                modality="text",
                normalizer_id=MTP_NORMALIZER_ID,
                capture_layout=OfflineCaptureLayout(
                    capture_method="dflash",
                    aux_feature=None,
                    last_hidden_feature="target_last_hidden_states",
                    passthrough=(
                        ("input_ids", "input_ids"),
                        ("loss_mask", "loss_mask"),
                    ),
                ),
                build_reader=partial(build_mtp_offline_reader, ALGORITHM_NAME),
                build_normalizer=build_mtp_offline_normalizer,
                build_collator=build_mtp_collator,
            ),
        ),
        server_streaming=(
            ServerStreamingProvider(
                modality="text",
                capture_method="dflash",
                target_representation="hidden_state",
                layout=ServerCaptureLayout(
                    aux_feature="hidden_states",
                    last_hidden_feature="target_last_hidden_states",
                    passthrough=(
                        ("input_ids", "input_ids", ()),
                        ("loss_mask", "loss_mask", ()),
                    ),
                ),
                build_collator=build_mtp_collator,
            ),
        ),
    )


__all__ = ["algorithm_providers", "algorithm_spec", "create_registration"]
