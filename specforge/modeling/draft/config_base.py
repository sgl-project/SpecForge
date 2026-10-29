"""Compatibility import; use :mod:`specforge.modeling.draft.config.base`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("specforge.modeling.draft.config.base")
