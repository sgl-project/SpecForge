"""SpecForge's torch_tpu backend as an out-of-tree plugin.

All TPU-only code lives in this package. Core SpecForge never imports it; the
``specforge.plugins`` entry point calls register(), which patches the core
seams listed in specforge_tpu.patches.
"""

from __future__ import annotations

import os

_registered = False


def register() -> None:
    global _registered
    if _registered or os.environ.get("SPECFORGE_DEVICE") != "tpu":
        return
    from specforge_tpu import patches

    patches.apply()
    _registered = True
