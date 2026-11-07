"""Deprecated: moved to :mod:`specforge.modeling.draft.layers`."""

import warnings

from specforge.modeling.draft.layers import (
    ColumnParallelLinear,
    ParallelLMHead,
    RowParallelLinear,
    VocabParallelEmbedding,
)

warnings.warn(
    "specforge.layers has moved to specforge.modeling.draft.layers; the old "
    "import path will be removed in a future release",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "VocabParallelEmbedding",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "ParallelLMHead",
]
