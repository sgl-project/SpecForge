"""Architecture protocols and auxiliary computations, composed from nodes."""

from .base import ArchitectureSpec
from .dflash import DFlashSpec
from .dflash2 import DFlash2Spec
from .domino import DominoSpec
from .dspark import DSparkSpec
from .dspine import DSpineSpec
from .eagle3 import Eagle3Spec
from .mtp import MTPSpec
from .peagle import PEagleSpec

__all__ = [
    "ArchitectureSpec",
    "DFlashSpec",
    "DFlash2Spec",
    "DominoSpec",
    "DSparkSpec",
    "DSpineSpec",
    "Eagle3Spec",
    "MTPSpec",
    "PEagleSpec",
]
