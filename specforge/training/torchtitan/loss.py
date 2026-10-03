"""SpecForge objectives consumed by TorchTitan's native backward step."""

from dataclasses import dataclass

import torch
from torchtitan.components.loss import BaseLoss


class SpecForgeObjectiveLoss(BaseLoss):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        algorithm: str = "dflash"

    def __init__(self, config: Config, *, compile_config=None):
        if config.algorithm not in {"dflash", "dflash2", "dspark"}:
            raise ValueError(f"Unsupported TorchTitan objective: {config.algorithm}")
        self.algorithm = config.algorithm

    def __call__(self, pred, labels, global_valid_tokens=None, **kwargs):
        # Titan's token count is useful for throughput, but sampled block losses
        # have a different denominator. Preparation supplies the exact optimizer
        # window denominator before Titan performs any forward/backward calls.
        del labels, global_valid_tokens, kwargs
        if isinstance(pred, torch.Tensor):
            # The last pipeline stage returns tensors only; carry the same
            # objective numerator and prepared denominator as two scalars.
            return pred[0] / pred[1].clamp_min(1e-30), {}
        _loss, _accuracy, metrics = pred
        normalizer = metrics["_torchtitan_normalizer"]
        numerator, _denominator = metrics["loss_terms"]
        return numerator / normalizer.clamp_min(1e-30), metrics
