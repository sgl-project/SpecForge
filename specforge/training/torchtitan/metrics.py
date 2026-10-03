"""Logger adaptation for sampled-block objectives in the native trainer."""

from typing import Any

from torchtitan.components.metrics import BaseLogger


class ObjectiveMetricLogger(BaseLogger):
    """Keep native metrics except a maximum derived from token denominators.

    Titan reconstructs each rank's mean loss from its valid-token count. That
    conversion is not defined for SpecForge's weighted sampled-block objective,
    and can be NaN on an empty CP slice. The reduced average objective remains
    correct. Omit the unsupported maximum rather than presenting a false one.
    """

    def __init__(self, delegate: BaseLogger):
        self.delegate = delegate

    def log(self, metrics: dict[str, Any], step: int) -> None:
        supported_metrics = {
            name: value
            for name, value in metrics.items()
            if name != "loss_metrics/global_max_loss"
        }
        self.delegate.log(supported_metrics, step)

    def close(self) -> None:
        self.delegate.close()
