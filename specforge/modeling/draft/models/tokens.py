"""Assemble owned or target-bound token modules without changing their names."""

import torch.nn.functional as F
from torch import Tensor, nn

from specforge.config.draft import DraftConfig

from ..bindings import BoundEmbedding, BoundLMHead


class LMHead(nn.Linear):
    """Owned vocabulary projection with the same candidate API as BoundLMHead."""

    def candidate_weights(self, token_ids: Tensor) -> Tensor:
        return F.embedding(token_ids, self.weight)


def build_token_modules(config: DraftConfig, *, device=None, dtype=None):
    """Return embedding/head modules; their owner performs initialization/tying."""
    options = dict(device=device, dtype=dtype)
    embedding = (
        BoundEmbedding(config.vocab_size, config.hidden_size, **options)
        if "embed_tokens" in config.target_bindings
        else nn.Embedding(config.vocab_size, config.hidden_size, **options)
    )
    head = (
        BoundLMHead(
            config.target_logit_transform,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            **options,
        )
        if "lm_head" in config.target_bindings
        else LMHead(config.hidden_size, config.draft_vocab_size, bias=False, **options)
    )
    return embedding, head
