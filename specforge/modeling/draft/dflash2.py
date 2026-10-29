"""Compatibility import; use :mod:`specforge.modeling.draft.models.dflash2`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "specforge.modeling.draft.models.dflash2"
)
