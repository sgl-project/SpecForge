"""Compatibility exports; new code imports draft.config."""

from .config.eagle3 import Eagle3Config, EagleConfig, EaglePretrainedConfig
from .config.peagle import PEagleConfig

__all__ = ["Eagle3Config", "EagleConfig", "EaglePretrainedConfig", "PEagleConfig"]
