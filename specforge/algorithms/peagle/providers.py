"""Built-in P-EAGLE registration and server-streaming providers."""

from __future__ import annotations

from specforge.algorithms.common.defaults import (
    empty_options,
    one_loss_token,
    online_needs_input_tools,
)
from specforge.algorithms.common.providers import (
    AlgorithmProviders,
    DraftConfigProvider,
    ModelProvider,
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
)
from specforge.algorithms.eagle3.data import build_server_collator

ALGORITHM_NAME = "peagle"
DRAFT_ARCHITECTURE = "PEagleDraftModel"


def build_step(wrapped_model, *, target_head=None, **_options):
    from specforge.training.strategies.base import PEagleTrainStrategy

    return PEagleTrainStrategy(wrapped_model, target_head=target_head)


def resume_contract(config, draft_model, training_model):
    from specforge.algorithms.model_providers import peagle_resume_contract

    return peagle_resume_contract(config, draft_model, training_model)


def build_draft(config, draft_config):
    from specforge.algorithms.model_providers import build_peagle_draft

    return build_peagle_draft(config, draft_config)


def build_training_model(config, draft_model, draft_config, target_config, tokenizer):
    from specforge.algorithms.model_providers import build_peagle_model

    return build_peagle_model(
        config,
        draft_model,
        draft_config,
        target_config,
        tokenizer,
    )


def resolve_capture_layers(config, draft_config, target_config):
    from specforge.algorithms.model_providers import resolve_eagle_capture_layers

    return resolve_eagle_capture_layers(config, draft_config, target_config)


def algorithm_spec() -> AlgorithmSpec:
    return AlgorithmSpec(
        name=ALGORITHM_NAME,
        draft=DraftRequirement(
            compatible_architectures={DRAFT_ARCHITECTURE},
            default_architecture=DRAFT_ARCHITECTURE,
            supported_overrides={"attention_layout", "num_hidden_layers"},
        ),
        feature_contracts=(
            FeatureContract(
                mode=FeatureMode.STREAMING,
                modality="text",
                required_tensors={
                    "input_ids",
                    "attention_mask",
                    "loss_mask",
                    "hidden_state",
                    "target",
                },
                optional_tensors={"lengths"},
                allowed_target_representations={"hidden_state"},
                default_target_representation="hidden_state",
            ),
        ),
        capabilities=AlgorithmCapabilities(
            attention_backends={"flex_attention"},
            required_batch_size=1,
            supports_vocab_mapping=True,
            allows_aux_layer_override=True,
        ),
    )


def algorithm_providers() -> AlgorithmProviders:
    return AlgorithmProviders(
        algorithm_name=ALGORITHM_NAME,
        step=StepProvider(
            build=build_step,
            options=empty_options,
            resume_contract=resume_contract,
            # Server capture stores the target's last hidden state.  The
            # consumer therefore injects a frozen target head before cutover.
            uses_external_target_head=True,
        ),
        model=ModelProvider(
            draft_config=DraftConfigProvider(
                architecture=DRAFT_ARCHITECTURE,
                target_defaults=TargetDerivedDraftDefaults(
                    model_type="llama",
                    num_hidden_layers=4,
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
        vocab_mapping_modes=frozenset({FeatureMode.STREAMING}),
    )


def create_registration():
    return make_registration(algorithm_spec(), algorithm_providers())


__all__ = ["algorithm_providers", "algorithm_spec", "create_registration"]
