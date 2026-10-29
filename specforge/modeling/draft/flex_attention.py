"""Compatibility import; use :mod:`specforge.modeling.draft.models.layers.attention.backends.flex`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "specforge.modeling.draft.models.layers.attention.backends.flex"
)
