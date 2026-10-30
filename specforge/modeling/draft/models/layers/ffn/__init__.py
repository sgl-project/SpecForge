"""Token-wise FFN component construction."""

from specforge.config.draft.component_registry import build_component


def build_ffn(spec, *, config, dense_factory):
    return build_component(
        "ffn", spec, "tokenwise", config=config, dense_factory=dense_factory
    )


def build_dense(*, spec, config, dense_factory):
    return dense_factory(config)
