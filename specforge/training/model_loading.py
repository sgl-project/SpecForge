"""Resolve draft metadata and preflight training kernels before model assembly."""


def resolve_draft_config(cfg, *, provider=None):
    from .canonical import resolve_config

    return resolve_config(cfg)


def preflight_draft_kernels(cfg, *, algorithm):
    if cfg.model.use_liger_kernel is True:
        from specforge.modeling.draft.layers.kernels import load_liger_kernels

        load_liger_kernels()


def draft_config_dict(cfg, *, provider=None):
    from specforge.config.draft.codec import encode_config

    return encode_config(resolve_draft_config(cfg))


def training_target_config(cfg):
    """Use the checkpoint reader's target metadata for canonical training."""
    from specforge.checkpoint.resolve import read_target_config

    return read_target_config(
        cfg.model.target_model_path, cache_dir=cfg.model.cache_dir
    )
