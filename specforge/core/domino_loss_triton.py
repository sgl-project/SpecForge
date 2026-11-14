"""Compatibility alias for specforge.losses._domino_triton."""

import sys

from specforge.losses import _domino_triton

# Share module state and preserve patches made through the old import path.
sys.modules[__name__] = _domino_triton
