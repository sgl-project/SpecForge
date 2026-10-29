"""Compatibility alias for the DSpine draft model."""

import sys

from .models import dspine as _implementation

sys.modules[__name__] = _implementation
