"""Compatibility import; use :mod:`specforge.modeling.draft.models.specialized.qwen3_5_mtp.model`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "specforge.modeling.draft.models.specialized.qwen3_5_mtp.model"
)
