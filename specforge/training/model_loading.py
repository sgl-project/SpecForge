"""Resolve draft metadata and preflight training kernels before model assembly."""


def resolve_draft_config(cfg):
    from .canonical import resolve_config

    return resolve_config(cfg)


def preflight_draft_kernels(cfg, *, algorithm):
    from specforge.modeling.draft.layers.mlp import EXPERT_KERNELS

    from .canonical import preflight_kernels

    if cfg.model.moe_experts_kernel not in EXPERT_KERNELS:
        raise ValueError(
            f"unknown model.moe_experts_kernel {cfg.model.moe_experts_kernel!r}; "
            f"expected one of {sorted(EXPERT_KERNELS)}"
        )
    preflight_kernels(cfg)


def draft_config_dict(cfg):
    from specforge.config.draft.codec import encode_config

    return encode_config(resolve_draft_config(cfg))


def training_target_config(cfg):
    """Use the checkpoint reader's target metadata for canonical training."""
    from specforge.checkpoint.resolve import read_target_config

    return read_target_config(
        cfg.model.target_model_path, cache_dir=cfg.model.cache_dir
    )
