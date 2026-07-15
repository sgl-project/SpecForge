"""Built-in DSpark registration and executable providers."""

from __future__ import annotations

from specforge.algorithms.common.defaults import (
    empty_options,
    no_missing_checkpoint_keys,
)
from specforge.algorithms.common.dflash_family_data import build_dspark_collator
from specforge.algorithms.common.providers import (
    AlgorithmProviders,
    DraftConfigProvider,
    ModelProvider,
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
)

ALGORITHM_NAME = "dspark"
DRAFT_ARCHITECTURE = "DSparkDraftModel"


def build_step(wrapped_model, *, target_head=None, **_options):
    del target_head
    from specforge.training.strategies.base import DSparkTrainStrategy

    return DSparkTrainStrategy(wrapped_model)


def resume_contract(_config, draft_model, training_model):
    """Persist resolved DSpark model, sampling, and objective semantics."""

    return {
        "dspark_draft_num_hidden_layers": int(draft_model.config.num_hidden_layers),
        "dspark_target_layer_ids": tuple(
            int(layer_id) for layer_id in draft_model.target_layer_ids
        ),
        "dspark_block_size": int(training_model.block_size),
        "dspark_mask_token_id": int(training_model.mask_token_id),
        "dspark_attention_backend": str(training_model.attention_backend),
        "dspark_num_anchors": int(training_model.num_anchors),
        "dspark_loss_decay_gamma": training_model.loss_decay_gamma,
        "dspark_ce_loss_alpha": float(training_model.dspark_ce_loss_alpha),
        "dspark_l1_loss_alpha": float(training_model.dspark_l1_loss_alpha),
        "dspark_confidence_head_alpha": float(
            training_model.dspark_confidence_head_alpha
        ),
    }


def build_draft(config, draft_config):
    from specforge.algorithms.model_providers import build_registered_draft

    return build_registered_draft(config, draft_config)


def build_training_model(config, draft_model, draft_config, target_config, tokenizer):
    from specforge.algorithms.model_providers import build_dspark_model

    return build_dspark_model(
        config,
        draft_model,
        draft_config,
        target_config,
        tokenizer,
    )


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
    return AlgorithmSpec(
        name=ALGORITHM_NAME,
        draft=DraftRequirement(
            compatible_architectures={DRAFT_ARCHITECTURE},
            default_architecture=DRAFT_ARCHITECTURE,
        ),
        feature_contracts=(
            FeatureContract(
                mode=FeatureMode.STREAMING,
                modality="text",
                required_tensors={
                    "input_ids",
                    "loss_mask",
                    "hidden_states",
                    "target_last_hidden_states",
                },
                allowed_target_representations={"hidden_state"},
                default_target_representation="hidden_state",
            ),
        ),
        capabilities=AlgorithmCapabilities(
            attention_backends={"eager", "sdpa", "flex_attention"},
        ),
    )


def algorithm_providers() -> AlgorithmProviders:
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
                expected_auto_map_model="dspark.DSparkDraftModel",
            ),
            build_draft=build_draft,
            build_training_model=build_training_model,
            resolve_capture_layers=resolve_capture_layers,
            minimum_loss_tokens=minimum_loss_tokens,
            needs_input_tools=needs_input_tools,
            default_dataloader_num_workers=8,
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
                build_collator=build_dspark_collator,
            ),
        ),
    )


def create_registration():
    return make_registration(algorithm_spec(), algorithm_providers())


__all__ = ["algorithm_providers", "algorithm_spec", "create_registration"]
