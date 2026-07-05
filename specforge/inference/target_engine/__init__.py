# coding=utf-8
"""Target engines: the backend-agnostic capture surface (TargetEngine + factory)."""

from .base import KNOWN_BACKENDS, TargetEngine
from .custom import CustomTargetEngine
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
from .hf import HFTargetEngine
from .sglang import SGLangTargetEngine
from .target_capture_policy import (
    CAPTURE_POLICIES,
    TARGET_CAPTURE_POLICIES,
    CapturePolicy,
    CaptureSpec,
    DFlashCapturePolicy,
    Eagle3CapturePolicy,
    TargetCaptureBatch,
    TargetCapturePolicy,
    TargetCaptureSpec,
    register_capture_policy,
    register_target_capture_policy,
    resolve_capture_policy,
    resolve_target_capture_policy,
)

__all__ = [
    "TargetEngine",
    "KNOWN_BACKENDS",
    "get_target_engine",
    "available_target_engines",
    # Target-capture policies (per-algorithm axis of the engine matrix)
    "TargetCaptureBatch",
    "TargetCaptureSpec",
    "TargetCapturePolicy",
    "CaptureSpec",
    "CapturePolicy",
    "Eagle3CapturePolicy",
    "DFlashCapturePolicy",
    "TARGET_CAPTURE_POLICIES",
    "CAPTURE_POLICIES",
    "register_target_capture_policy",
    "resolve_target_capture_policy",
    "register_capture_policy",
    "resolve_capture_policy",
    # Generic per-backend engines (policy-parameterized)
    "HFTargetEngine",
    "SGLangTargetEngine",
    "CustomTargetEngine",
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
]

# NOTE: the DFlash engines (dflash_target_model) are intentionally NOT eagerly
# imported here — that module imports sglang internals unconditionally, and this
# package must stay importable without the pinned sglang. Import them from the
# submodule, or via get_target_engine(strategy="dflash", ...).
