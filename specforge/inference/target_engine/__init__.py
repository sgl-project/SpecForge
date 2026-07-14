# coding=utf-8
"""Policy-driven target engines and their single strategy/backend factory."""

from .base import KNOWN_BACKENDS, TargetEngine
from .custom import CustomTargetEngine
from .factory import available_target_engines, get_target_engine
from .hf import HFTargetEngine
from .sglang import SGLangTargetEngine
from .target_capture_policy import (
    TARGET_CAPTURE_POLICIES,
    DFlashCapturePolicy,
    DFlashTargetOutput,
    Eagle3CapturePolicy,
    Eagle3TargetOutput,
    TargetCaptureBatch,
    TargetCapturePolicy,
    TargetCaptureSpec,
    register_target_capture_policy,
    resolve_target_capture_policy,
)

__all__ = [
    "TargetEngine",
    "KNOWN_BACKENDS",
    "get_target_engine",
    "available_target_engines",
    "HFTargetEngine",
    "SGLangTargetEngine",
    "CustomTargetEngine",
    "TargetCaptureBatch",
    "TargetCaptureSpec",
    "TargetCapturePolicy",
    "Eagle3CapturePolicy",
    "DFlashCapturePolicy",
    "Eagle3TargetOutput",
    "DFlashTargetOutput",
    "TARGET_CAPTURE_POLICIES",
    "register_target_capture_policy",
    "resolve_target_capture_policy",
]
