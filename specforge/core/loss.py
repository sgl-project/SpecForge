"""Compatibility alias for specforge.losses.soft_cross_entropy."""

import sys

from specforge.losses import soft_cross_entropy

# Share module state and preserve patches made through the old import path.
sys.modules[__name__] = soft_cross_entropy
