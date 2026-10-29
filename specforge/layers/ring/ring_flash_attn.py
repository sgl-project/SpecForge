"""Compatibility import; use :mod:`specforge.modeling.draft.models.layers.attention.parallel.ring`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "specforge.modeling.draft.models.layers.attention.parallel.ring"
)
