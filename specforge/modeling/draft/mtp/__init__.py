# coding=utf-8
"""MTP draft architectures: one registered module per model family."""

from specforge.modeling.draft.mtp.base import MTPDraftModel
from specforge.modeling.draft.mtp.qwen3_5 import Qwen3_5MTPDraftModel

__all__ = ["MTPDraftModel", "Qwen3_5MTPDraftModel"]
