"""Assemble owned or target-bound token modules without changing their names."""

from torch import nn

from specforge.config.draft import DraftConfig

from ..bindings import BoundEmbedding, BoundLMHead


def build_token_modules(
    config: DraftConfig, *, padding_idx=None, device=None, dtype=None
):
    """Return embedding/head modules; their owner performs initialization/tying."""
    options = dict(device=device, dtype=dtype)
    embedding = (
        BoundEmbedding(config.vocab_size, config.hidden_size, **options)
        if "embed_tokens" in config.target_bindings
        else nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=padding_idx, **options
        )
    )
    head = (
        BoundLMHead(
            config.target_logit_transform,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            **options,
        )
        if "lm_head" in config.target_bindings
        else nn.Linear(
            config.hidden_size, config.draft_vocab_size, bias=False, **options
        )
    )
    return embedding, head
