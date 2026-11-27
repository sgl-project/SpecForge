"""Assemble owned or target-bound token modules without changing their names."""

from torch import nn

from specforge.config.draft import DraftConfig

from ..bindings import BoundEmbedding, BoundLMHead


def build_token_modules(
    config: DraftConfig, *, padding_idx=None, device=None, dtype=None
):
    """Return embedding/head modules; their owner performs initialization/tying.

    Bound modules hold the target's tables at the architecture's bound table
    width; owned modules use the draft's hidden size.
    """
    options = dict(device=device, dtype=dtype)
    bound_width = config.spec_config.bound_table_width(config.hidden_size)
    embedding = (
        BoundEmbedding(config.vocab_size, bound_width, **options)
        if "embed_tokens" in config.target_bindings
        else nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=padding_idx, **options
        )
    )
    head = (
        BoundLMHead(
            config.target_logit_transform,
            vocab_size=config.vocab_size,
            hidden_size=bound_width,
            **options,
        )
        if "lm_head" in config.target_bindings
        else nn.Linear(
            config.hidden_size, config.draft_vocab_size, bias=False, **options
        )
    )
    return embedding, head


def build_width_adapters(config: DraftConfig, *, device=None, dtype=None):
    """Return (input_adapter, output_adapter) between bound tables and the draft.

    Each is a bias-free Linear when its table is bound at a width other than
    hidden_size, and Identity otherwise, so equal-width drafts gain no state.
    """
    width = config.spec_config.bound_table_width(config.hidden_size)

    def adapter(table, in_features, out_features):
        if table not in config.target_bindings or width == config.hidden_size:
            return nn.Identity()
        return nn.Linear(
            in_features, out_features, bias=False, device=device, dtype=dtype
        )

    return (
        adapter("embed_tokens", width, config.hidden_size),
        adapter("lm_head", config.hidden_size, width),
    )
