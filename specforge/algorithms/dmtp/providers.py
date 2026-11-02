"""DMTP reuses DFlash construction and loss, with a distinct feature contract."""

from dataclasses import replace

from specforge.algorithms.common.providers import make_registration
from specforge.algorithms.dflash import providers as dflash

from . import data

ARCHITECTURE = "DMTPDraftModel"


def build_step(wrapped_model, **_):
    from .model import DMTPTrainStrategy

    return DMTPTrainStrategy(wrapped_model)


def build_training_model(config, draft_model, draft_config, target_config, tokenizer):
    from specforge.algorithms.model_providers import _build_dflash_family_model

    from .model import OnlineDMTPModel

    return _build_dflash_family_model(
        config,
        draft_model,
        tokenizer,
        lambda common: OnlineDMTPModel(
            **common,
            loss_type=config.training.loss_type,
            dpace_alpha=config.training.dpace_alpha,
            lk_loss_type=config.training.lk_loss_type,
            kl_scale=config.training.kl_scale,
            kl_decay=config.training.kl_decay,
            teacher_metrics=config.training.dflash_teacher_metrics,
            fused_logits_loss=config.training.fused_logits_loss,
        ),
    )


def select_layout(config):
    return replace(
        dflash.select_server_capture_layout(config), mtp_hidden_feature="mtp_hidden"
    )


def resume_contract(config, draft_model, training_model):
    return {
        **{
            key.replace("dflash_", "dmtp_", 1): value
            for key, value in dflash.resume_contract(
                config, draft_model, training_model
            ).items()
        },
        "dmtp_alignment": "anchor_post_norm_v1",
        "dmtp_context_offset": -1,
    }


def create_registration():
    base = dflash.algorithm_spec()
    contracts = tuple(
        replace(
            contract,
            required_tensors=set(data.FEATURES),
            storage=(
                replace(
                    contract.storage,
                    required_tensors=set(data.FEATURES),
                    normalizer=data.NORMALIZER_ID,
                )
                if contract.storage
                else None
            ),
        )
        for contract in base.feature_contracts
    )
    spec = replace(
        base,
        name="dmtp",
        feature_contracts=contracts,
        draft=replace(
            base.draft,
            compatible_architectures={ARCHITECTURE},
            default_architecture=ARCHITECTURE,
        ),
    )
    providers = dflash.algorithm_providers()
    providers = replace(
        providers,
        algorithm_name="dmtp",
        step=replace(providers.step, build=build_step, resume_contract=resume_contract),
        model=replace(
            providers.model,
            build_training_model=build_training_model,
            loss_mask_filter=data.loss_mask_filter,
            draft_config=replace(
                providers.model.draft_config,
                architecture=ARCHITECTURE,
                compatible_architectures={ARCHITECTURE},
                expected_auto_map_model="dmtp.DMTPDraftModel",
            ),
        ),
        offline=tuple(
            replace(
                item,
                normalizer_id=data.NORMALIZER_ID,
                capture_layout=None,
                build_reader=data.build_reader,
                build_normalizer=data.build_normalizer,
                build_collator=data.build_collator,
            )
            for item in providers.offline
        ),
        server_streaming=tuple(
            replace(
                item,
                layout=replace(item.layout, mtp_hidden_feature="mtp_hidden"),
                select_layout=select_layout,
                build_collator=data.build_collator,
            )
            for item in providers.server_streaming
        ),
    )
    return make_registration(spec, providers)
