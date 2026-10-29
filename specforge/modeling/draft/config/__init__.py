"""Public draft configuration API. Parsing does not construct model parameters."""

from .auto import AutoDraftConfig
from .base import DraftConfig
from .dflash import DFlashConfig
from .dflash2 import DFlash2Config
from .domino import DominoConfig
from .dspark import DSparkConfig
from .eagle3 import Eagle3Config, EagleConfig, EaglePretrainedConfig
from .mtp import Qwen3_5MTPConfig
from .peagle import PEagleConfig

DraftConfigClass = AutoDraftConfig

__all__ = [
    "AutoDraftConfig",
    "DraftConfigClass",
    "DraftConfig",
    "DFlashConfig",
    "DFlash2Config",
    "DominoConfig",
    "DSparkConfig",
    "Eagle3Config",
    "EagleConfig",
    "EaglePretrainedConfig",
    "Qwen3_5MTPConfig",
    "PEagleConfig",
]
