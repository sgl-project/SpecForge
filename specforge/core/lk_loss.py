"""Compatibility alias for specforge.losses.lk."""

import sys

from specforge.losses import lk

# Share module state and preserve patches made through the old import path.
sys.modules[__name__] = lk
