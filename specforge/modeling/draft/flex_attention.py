"""Deprecated: moved to :mod:`specforge.modeling.draft.layers.attention.backends.flex_attention`.

``generate_eagle3_mask`` moved to :mod:`specforge.modeling.draft.eagle3_mask`.
This module re-exports both; patch the new modules, not this one.
"""

import warnings

from specforge.modeling.draft.eagle3_mask import generate_eagle3_mask
from specforge.modeling.draft.layers.attention.backends.flex_attention import (
    WrappedCreateBlockMask,
    WrappedFlexAttention,
    compile_friendly_create_block_mask,
    compile_friendly_flex_attention,
)

warnings.warn(
    "specforge.modeling.draft.flex_attention has moved to "
    "specforge.modeling.draft.layers.attention.backends.flex_attention "
    "(generate_eagle3_mask: specforge.modeling.draft.eagle3_mask); the old "
    "import path will be removed in a future release",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "WrappedCreateBlockMask",
    "WrappedFlexAttention",
    "compile_friendly_create_block_mask",
    "compile_friendly_flex_attention",
    "generate_eagle3_mask",
]
