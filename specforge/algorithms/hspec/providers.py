"""Built-in H-Spec registration and executable providers."""

from __future__ import annotations

from functools import partial

from specforge.algorithms.common.defaults import (
    empty_options,
    no_missing_checkpoint_keys,
)
from specforge.algorithms.common.hidden_states_data import (
    HSPEC_NORMALIZER_ID,
    build_hspec_collator,
    build_hspec_offline_normalizer,
    build_hspec_offline_reader,
)
from specforge.algorithms.common.providers import (
    AlgorithmProviders,
    DraftConfigProvider,
    ModelProvider,
    OfflineCaptureLayout,
    OfflineDataProvider,
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

ALGORITHM_NAME = "hspec"
DRAFT_ARCHITECTURE = "HSpecDraftModel"
COMPATIBLE_DRAFT_ARCHITECTURES = frozenset({DRAFT_ARCHITECTURE})

_OFFLINE_FEATURES = (
    "input_ids",
    "loss_mask",
    "prefix_masks",
    "hidden_states",
    "target_last_hidden_states",
    "selected_target_k",
    "selected_target_v",
)


def build_step(wrapped_model, *, target_head=None, **_options):
    del target_head
    from specforge.training.strategies.base import HSpecTrainStrategy

    return HSpecTrainStrategy(wrapped_model)


def resume_contract(_config, draft_model, training_model):
    """Persist resolved H-Spec model, sampling, and objective semantics."""

    return {
        "hspec_draft_num_hidden_layers": int(draft_model.config.num_hidden_layers),
        "hspec_target_layer_ids": tuple(
            int(layer_id) for layer_id in draft_model.target_layer_ids
        ),
        "hspec_target_kv_layer_ids": tuple(
            int(layer_id) for layer_id in draft_model.target_kv_layer_ids
        ),
        "hspec_block_pattern": tuple(draft_model.block_pattern),
        "hspec_block_size": int(training_model.block_size),
        "hspec_mask_token_id": int(training_model.mask_token_id),
        "hspec_attention_backend": str(training_model.attention_backend),
        "hspec_num_anchors": int(training_model.num_anchors),
        "hspec_loss_decay_gamma": training_model.loss_decay_gamma,
        "hspec_ce_loss_alpha": float(training_model.dspark_ce_loss_alpha),
        "hspec_l1_loss_alpha": float(training_model.dspark_l1_loss_alpha),
        "hspec_confidence_head_alpha": float(
            training_model.dspark_confidence_head_alpha
        ),
    }


def build_draft(config, draft_config):
    from specforge.algorithms.model_providers import build_registered_draft

    return build_registered_draft(config, draft_config)


def build_training_model(config, draft_model, draft_config, target_config, tokenizer):
    from specforge.algorithms.model_providers import build_hspec_model

    return build_hspec_model(
        config,
        draft_model,
        draft_config,
        target_config,
        tokenizer,
    )


def resolve_capture_layers(config, draft_config, target_config):
    from specforge.algorithms.model_providers import resolve_hspec_capture_layers

    return resolve_hspec_capture_layers(config, draft_config, target_config)


def minimum_loss_tokens(config, draft_config):
    from specforge.algorithms.model_providers import dflash_min_loss_tokens

    return dflash_min_loss_tokens(config, draft_config)


def needs_input_tools(config, draft_model):
    from specforge.algorithms.model_providers import dflash_needs_input_tools

    return dflash_needs_input_tools(config, draft_model)


def algorithm_spec() -> AlgorithmSpec:
    ready = set(_OFFLINE_FEATURES)
    return AlgorithmSpec(
        name=ALGORITHM_NAME,
        draft=DraftRequirement(
            compatible_architectures=COMPATIBLE_DRAFT_ARCHITECTURES,
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
                    normalizer=HSPEC_NORMALIZER_ID,
                ),
            ),
        ),
        capabilities=AlgorithmCapabilities(
            # split-source attention is wired for eager/sdpa only; flex
            # attention still routes through the fused cat path.
            attention_backends={"eager", "sdpa"},
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
                compatible_architectures=COMPATIBLE_DRAFT_ARCHITECTURES,
                expected_auto_map_model="hspec.HSpecDraftModel",
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
                normalizer_id=HSPEC_NORMALIZER_ID,
                capture_layout=OfflineCaptureLayout(
                    capture_method="hspec",
                    aux_feature="hidden_states",
                    last_hidden_feature="target_last_hidden_states",
                    passthrough=(
                        ("input_ids", "input_ids"),
                        ("loss_mask", "loss_mask"),
                        ("prefix_masks", "prefix_masks"),
                        ("selected_target_k", "selected_target_k"),
                        ("selected_target_v", "selected_target_v"),
                    ),
                ),
                build_reader=partial(build_hspec_offline_reader, ALGORITHM_NAME),
                build_normalizer=build_hspec_offline_normalizer,
                build_collator=build_hspec_collator,
            ),
        ),
    )


def create_registration():
    return make_registration(algorithm_spec(), algorithm_providers())


__all__ = ["algorithm_providers", "algorithm_spec", "create_registration"]
