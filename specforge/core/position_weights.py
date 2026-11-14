"""Compatibility alias for specforge.losses.position_weights."""

import sys

from specforge.losses import position_weights

# Share module state and preserve patches made through the old import path.
sys.modules[__name__] = position_weights
