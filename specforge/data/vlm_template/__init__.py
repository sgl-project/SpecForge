# Copyright 2024 SpecForge Authors
# Licensed under the Apache License, Version 2.0

"""
VLM Template module for SpecForge.

This module provides template classes for different VLM models, following
the design pattern from ms-swift. Each template encapsulates model-specific
encoding and collation logic.
"""

from specforge.data.vlm_template.base import VLMTemplate
from specforge.data.vlm_template.minicpm import MiniCPMVTemplate
from specforge.data.vlm_template.qwen import QwenVLTemplate

__all__ = [
    "VLMTemplate",
    "MiniCPMVTemplate",
    "QwenVLTemplate",
]
