"""Compatibility import; use :mod:`specforge.modeling.draft.models.layers.attention.backends.flex_options`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "specforge.modeling.draft.models.layers.attention.backends.flex_options"
)
