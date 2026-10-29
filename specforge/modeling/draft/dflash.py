"""Compatibility import; use :mod:`specforge.modeling.draft.models.dflash`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "specforge.modeling.draft.models.dflash"
)
