"""Built-in DSpark registration and executable providers."""

from __future__ import annotations

from functools import partial

from specforge.algorithms.common.defaults import (
    empty_options,
    no_missing_checkpoint_keys,
)
from specforge.algorithms.common.dflash_family_data import (
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

ALGORITHM_NAME = "dspark"
DRAFT_ARCHITECTURE = "DSparkDraftModel"
COMPATIBLE_DRAFT_ARCHITECTURES = frozenset({DRAFT_ARCHITECTURE})


def build_step(wrapped_model, *, target_head=None, **_options):
    del target_head
    from specforge.training.strategies.base import DSparkTrainStrategy

    return DSparkTrainStrategy(wrapped_model)


def resume_contract(_config, draft_model, training_model):
    """Persist resolved DSpark model, sampling, and objective semantics."""

    contract = {
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
    # Only recorded when the run actually prunes. The trainer treats a contract
    # key that a checkpoint does not carry as fatal, so adding this
    # unconditionally would make every DSpark checkpoint written before this
    # feature unresumable -- for a field that, unpruned, could only ever hold
    # vocab_size. It catches a changed K before any weight loads, and nothing
    # more: two mappings of the same size select different tokens without
    # disagreeing here. Which tokens is checked where the buffers actually meet,
    # in _reject_conflicting_vocab_mapping, since the contract is bound before
    # the offline mapping is installed and cannot see it.
    if getattr(draft_model, "use_draft_vocab", False):
        contract["dspark_draft_vocab_size"] = int(draft_model.draft_vocab_size)
    return contract


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
    ready = {
        "input_ids",
        "loss_mask",
        "hidden_states",
        "target_last_hidden_states",
    }
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
            supports_vocab_mapping=True,
            # DFlash-family drafts register t2d/d2t only when they prune, so an
            # unpruned run has nowhere to put a mapping. See _validate_vocab_mapping.
            keeps_vocab_buffers_when_unpruned=False,
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
                expected_auto_map_model="dspark.DSparkDraftModel",
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
                build_reader=partial(build_dspark_offline_reader, ALGORITHM_NAME),
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
        vocab_mapping_modes=frozenset({FeatureMode.OFFLINE, FeatureMode.STREAMING}),
    )


def create_registration():
    return make_registration(algorithm_spec(), algorithm_providers())


__all__ = ["algorithm_providers", "algorithm_spec", "create_registration"]
