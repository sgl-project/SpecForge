from .base import KNOWN_BACKENDS, TargetEngine
from .eagle3_target_model import (
    CustomEagle3TargetEngine,
    CustomEagle3TargetModel,
    Eagle3TargetEngine,
    Eagle3TargetModel,
    HFEagle3TargetEngine,
    HFEagle3TargetModel,
    SGLangEagle3TargetEngine,
    SGLangEagle3TargetModel,
    SGLangServerEagle3TargetEngine,
    get_eagle3_target_model,
)
from .factory import available_target_engines, get_target_engine
from .target_head import TargetHead

__all__ = [
    # Generic (Phase B) surface
    "TargetEngine",
    "KNOWN_BACKENDS",
    "get_target_engine",
    "available_target_engines",
    # EAGLE3 engines
    "Eagle3TargetEngine",
    "SGLangEagle3TargetEngine",
    "HFEagle3TargetEngine",
    "CustomEagle3TargetEngine",
    "SGLangServerEagle3TargetEngine",
    "get_eagle3_target_model",
    # Back-compat aliases (pre-Phase-B names)
    "Eagle3TargetModel",
    "SGLangEagle3TargetModel",
    "HFEagle3TargetModel",
    "CustomEagle3TargetModel",
    "TargetHead",
]

# NOTE: the DFlash engines (dflash_target_model) are intentionally NOT eagerly
# imported here — that module imports sglang internals unconditionally, and this
# package must stay importable without the pinned sglang (see factory._resolve_loader
# and eagle3_target_model's module docstring). Import them from the submodule, or
# via get_target_engine(strategy="dflash", ...).
