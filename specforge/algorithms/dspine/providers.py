"""DSpine uses the existing target-feature capture and training lifecycle."""

from dataclasses import asdict, replace
from functools import partial

from specforge.algorithms.common.hidden_states_data import build_dspark_offline_reader
from specforge.algorithms.common.providers import (
    TargetDerivedDraftDefaults,
    make_registration,
)
from specforge.algorithms.dflash.providers import apply_draft_overrides
from specforge.algorithms.dspark import providers as dspark


def build_step(wrapped_model, **_options):
    from .strategy import DSpineTrainStrategy

    return DSpineTrainStrategy(wrapped_model)


def build_draft(config, draft_config):
    from specforge.algorithms.model_providers import build_registered_draft

    if config.model.use_liger_kernel:
        from specforge.algorithms.model_providers import build_dflash_draft
        from specforge.modeling.draft.dflash_kernels import load_liger_dflash_kernels

        return build_dflash_draft(config, draft_config, load_liger_dflash_kernels())
    return build_registered_draft(config, draft_config)


def build_training_model(config, draft_model, draft_config, target_config, tokenizer):
    from specforge.algorithms.model_providers import _build_dflash_family_model

    from .model import OnlineDSpineModel

    return _build_dflash_family_model(
        config, draft_model, tokenizer, lambda common: OnlineDSpineModel(**common)
    )


def populate_target_defaults(payload, target_config, config):
    from specforge.algorithms.model_providers import populate_dflash_generated_config
    from specforge.modeling.draft.dflash import build_target_layer_ids

    populate_dflash_generated_config(payload, target_config, config)
    payload["dflash_config"]["target_layer_ids"] = build_target_layer_ids(
        payload["num_target_layers"], payload["num_hidden_layers"]
    )


def resume_contract(_config, draft_model, training_model):
    return {
        "dspine_options": tuple(sorted(asdict(draft_model.dspine_config).items())),
        "dspine_block_size": draft_model.block_size,
        "dspine_target_layer_ids": tuple(draft_model.target_layer_ids),
        "dspine_num_hidden_layers": len(draft_model.layers),
        "dspine_num_anchors": training_model.num_anchors,
        "dspine_mask_token_id": training_model.mask_token_id,
        "dspine_attention_backend": training_model.attention_backend,
        "dspine_loss_decay_gamma": training_model.loss_decay_gamma,
    }


def create_registration():
    source = dspark.create_registration()
    architectures = frozenset({"DSpineDraftModel"})
    spec = replace(
        source.spec,
        name="dspine",
        draft=replace(
            source.spec.draft,
            compatible_architectures=architectures,
            default_architecture="DSpineDraftModel",
            supported_overrides={"num_hidden_layers", "block_size"},
        ),
    )
    providers = source.providers
    model = replace(
        providers.model,
        draft_config=replace(
            providers.model.draft_config,
            architecture="DSpineDraftModel",
            compatible_architectures=architectures,
            expected_auto_map_model="dspine.DSpineDraftModel",
            target_defaults=TargetDerivedDraftDefaults(
                model_type="qwen3",
                num_hidden_layers=5,
                populate=populate_target_defaults,
            ),
            apply_overrides=apply_draft_overrides,
        ),
        build_draft=build_draft,
        build_training_model=build_training_model,
    )
    return make_registration(
        spec,
        replace(
            providers,
            algorithm_name="dspine",
            model=model,
            step=replace(
                providers.step, build=build_step, resume_contract=resume_contract
            ),
            offline=tuple(
                replace(
                    provider,
                    build_reader=partial(build_dspark_offline_reader, "dspine"),
                )
                for provider in providers.offline
            ),
        ),
    )
