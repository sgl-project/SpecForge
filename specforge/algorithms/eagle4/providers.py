"""Built-in EAGLE-4 registration. Capture, TTT, and the train strategy are EAGLE3's."""

from __future__ import annotations

from specforge.algorithms.common.defaults import (
    one_loss_token,
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
from specforge.algorithms.eagle3.data import (
    NORMALIZER_ID,
    build_offline_collator,
    build_offline_normalizer,
    build_offline_reader,
    build_server_collator,
)
from specforge.algorithms.eagle3.providers import (
    allowed_missing_checkpoint_keys,
    build_step,
    build_training_model,
    resolve_capture_layers,
    resume_contract as eagle3_resume_contract,
    step_options,
)

ALGORITHM_NAME = "eagle4"
DRAFT_ARCHITECTURE = "LlamaForCausalLMEagle4"


def build_draft(config, draft_config):
    from specforge.algorithms.model_providers import build_eagle4_draft

    return build_eagle4_draft(config, draft_config)


def resume_contract(config, draft_model, training_model):
    """Same resolved TTT settings as EAGLE3, keyed for an eagle4 run."""

    inner = eagle3_resume_contract(config, draft_model, training_model)
    return {
        (
            f"eagle4_{key[len('eagle3_'):]}"
            if key.startswith("eagle3_")
            else key
        ): value
        for key, value in inner.items()
    }


def algorithm_spec() -> AlgorithmSpec:
    required = {
        "input_ids",
        "attention_mask",
        "loss_mask",
        "hidden_state",
        "target",
    }
    return AlgorithmSpec(
        name=ALGORITHM_NAME,
        draft=DraftRequirement(
            compatible_architectures={DRAFT_ARCHITECTURE},
            default_architecture=DRAFT_ARCHITECTURE,
            supported_overrides={"attention_layout", "num_hidden_layers"},
            fixed_override_values=(("num_hidden_layers", 1),),
        ),
        feature_contracts=(
            FeatureContract(
                mode=FeatureMode.OFFLINE,
                modality="text",
                required_tensors=required,
                optional_tensors={"position_ids"},
                allowed_target_representations={"hidden_state"},
                default_target_representation="hidden_state",
                storage=OfflineStorageContract(
                    format="specforge_hidden_states_v1",
                    required_tensors={
                        "input_ids",
                        "loss_mask",
                        "hidden_state",
                        "aux_hidden_state",
                    },
                    normalizer=NORMALIZER_ID,
                ),
            ),
            FeatureContract(
                mode=FeatureMode.STREAMING,
                modality="text",
                required_tensors=required,
                allowed_target_representations={"hidden_state"},
                default_target_representation="hidden_state",
            ),
        ),
        capabilities=AlgorithmCapabilities(
            attention_backends={"sdpa", "flex_attention", "fa", "usp"},
            supports_compact_teacher=True,
            supports_trim_loss_positions=True,
            supports_vocab_mapping=True,
            allows_aux_layer_override=True,
        ),
    )


def algorithm_providers() -> AlgorithmProviders:
    return AlgorithmProviders(
        algorithm_name=ALGORITHM_NAME,
        step=StepProvider(
            build=build_step,
            options=step_options,
            resume_contract=resume_contract,
            allowed_missing_checkpoint_keys=allowed_missing_checkpoint_keys,
            uses_external_target_head=True,
        ),
        model=ModelProvider(
            draft_config=DraftConfigProvider(
                architecture=DRAFT_ARCHITECTURE,
                target_defaults=TargetDerivedDraftDefaults(
                    model_type="llama",
                    num_hidden_layers=1,
                    draft_vocab_size=32000,
                ),
            ),
            build_draft=build_draft,
            build_training_model=build_training_model,
            resolve_capture_layers=resolve_capture_layers,
            minimum_loss_tokens=one_loss_token,
            needs_input_tools=online_needs_input_tools,
            default_dataloader_num_workers=4,
            allow_missing_warm_start_embedding=True,
        ),
        offline=(
            OfflineDataProvider(
                modality="text",
                normalizer_id=NORMALIZER_ID,
                capture_layout=OfflineCaptureLayout(
                    capture_method="eagle3",
                    aux_feature="aux_hidden_state",
                    last_hidden_feature="hidden_state",
                    passthrough=(
                        ("input_ids", "input_ids"),
                        ("loss_mask", "loss_mask"),
                    ),
                ),
                build_reader=build_offline_reader,
                build_normalizer=build_offline_normalizer,
                build_collator=build_offline_collator,
            ),
        ),
        server_streaming=(
            ServerStreamingProvider(
                modality="text",
                capture_method="eagle3",
                target_representation="hidden_state",
                layout=ServerCaptureLayout(
                    aux_feature="hidden_state",
                    last_hidden_feature="target",
                    passthrough=(
                        ("input_ids", "input_ids", ()),
                        ("loss_mask", "loss_mask", ()),
                    ),
                    attention_mask_feature="attention_mask",
                ),
                build_collator=build_server_collator,
            ),
        ),
        vocab_mapping_modes=frozenset({FeatureMode.OFFLINE, FeatureMode.STREAMING}),
    )


def create_registration():
    return make_registration(algorithm_spec(), algorithm_providers())


__all__ = ["algorithm_providers", "algorithm_spec", "create_registration"]
