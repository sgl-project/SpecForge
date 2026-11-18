"""Target logit rules represented entirely by the installed head weight."""

from abc import ABC, abstractmethod

from torch import Tensor, nn

from specforge.config.draft.logits import TargetLogitSpec


class TargetLogitTransform(nn.Module, ABC):
    """Stateless weight transforms shared by full, candidate and fused scoring.

    All settings live in config; implementations must not own parameters or
    buffers. Output-only transforms such as softcap need a separate execution
    contract, since optimized scoring paths project directly through weight.
    """

    def __init__(self, config: TargetLogitSpec):
        super().__init__()
        self.config = config

    @abstractmethod
    def transform_weight_(self, weight: Tensor) -> None:
        """Modify a fresh copy in training dtype, once per target-weight load."""
