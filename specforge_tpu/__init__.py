"""SpecForge's torch_tpu backend as an out-of-tree plugin.

All TPU-only code lives in this package. Core SpecForge never imports it; the
``specforge.plugins`` entry point calls register(), which patches the core
seams listed in specforge_tpu.patches.
"""

from __future__ import annotations

import importlib.util
import os

_registered = False


def tpu_selected() -> bool:
    """SPECFORGE_DEVICE=tpu, or unset with torch_tpu installed."""
    device = os.environ.get("SPECFORGE_DEVICE")
    if device:
        return device == "tpu"
    return importlib.util.find_spec("torch_tpu") is not None


def register() -> None:
    global _registered
    if _registered or not tpu_selected():
        return
    _registered = True
    # Children (torchrun ranks, services) inherit the choice.
    os.environ["SPECFORGE_DEVICE"] = "tpu"
    from specforge_tpu import patches

    patches.apply()
