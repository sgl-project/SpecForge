"""Compatibility alias for specforge.losses.domino."""

import sys

from specforge.losses import domino

# Share module state and preserve patches made through the old import path.
sys.modules[__name__] = domino
