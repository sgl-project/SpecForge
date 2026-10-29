# coding=utf-8
"""MTP draft architectures: one registered module per model family."""

from specforge.modeling.draft.models.mtp import MTPDraftModel
from specforge.modeling.draft.models.specialized.qwen3_5_mtp.model import (
    Qwen3_5MTPDraftModel,
)

__all__ = ["MTPDraftModel", "Qwen3_5MTPDraftModel"]
