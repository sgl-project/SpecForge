"""DSpine schedule and optimizer-window loss contract."""

from __future__ import annotations

from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import (
    DFlashTrainStrategy,
    StepContext,
    StepOutput,
    _cpu_max_valid_anchors,
)


class DSpineTrainStrategy(DFlashTrainStrategy):
    name = "dspine"
    required_features = {
        "input_ids",
        "hidden_states",
        "loss_mask",
        "target_last_hidden_states",
    }

    def forward_loss(
        self, batch: TrainBatch, ctx: StepContext | None = None
    ) -> StepOutput:
        self.validate_batch(batch)
        tensors = batch.tensors
        inputs = {
            name: tensors[name].to(self._device(), non_blocking=True)
            for name in self.required_features
        }
        loss, accuracy, metrics = self.dflash_model(
            **inputs,
            max_valid_anchors=_cpu_max_valid_anchors(tensors["loss_mask"]),
            global_step=ctx.global_step if ctx is not None else 0,
            total_steps=ctx.total_steps if ctx is not None else None,
            collect_detailed_metrics=(
                ctx.collect_detailed_metrics if ctx is not None else True
            ),
        )
        return StepOutput(
            loss=loss,
            metrics={
                "accuracy": accuracy.detach(),
                "accuracy_denom": metrics["accuracy_denom"],
            },
            ratio_metrics=metrics["ratio_metrics"],
            loss_terms=metrics["loss_terms"],
        )
