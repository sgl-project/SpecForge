"""Compatibility alias for the shared numerical loss library."""

import sys

from specforge.losses import fused

# Preserve one module object, including lazy kernel state and existing patches.
sys.modules[__name__] = fused
