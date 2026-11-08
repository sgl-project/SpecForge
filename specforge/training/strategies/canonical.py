"""Adapt additive objective results to the existing distributed trainer."""

from dataclasses import replace

import torch
import torch.distributed as dist

from .base import DraftTrainStrategy, StepContext, StepOutput


class CanonicalTrainStrategy(DraftTrainStrategy):
    def __init__(self, model, *, name, objective, required_features):
        self.model = model
        self.name = name
        self.objective = objective
        self.required_features = set(required_features)
        seen = set()
        self._duplicate_parameters = set()
        for key, value in model.draft_model.named_parameters(remove_duplicate=False):
            if id(value) in seen:
                self._duplicate_parameters.add(key)
            seen.add(id(value))
        self._alignment_window = None
        self._alignment_ce = None

    def trainable_module(self):
        return self.model.draft_model

    def checkpoint_state_filter(self, state_dict):
        prefix = "draft_model."
        return {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
            and key[len(prefix) :] not in self._duplicate_parameters
        }

    def _forward(self, batch, ctx, alignment_ce=None):
        self.validate_batch(batch)
        device = next(self.model.parameters()).device
        tensors = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.tensors.items()
        }
        result = self.model(
            tensors,
            step=ctx.global_step,
            total_steps=ctx.total_steps,
            alignment_ce=alignment_ce,
        )
        return StepOutput(
            loss=result.loss,
            metrics={},
            ratio_metrics=result.ratios,
            loss_terms=(result.numerator, result.denominator),
        )

    def forward_loss(self, batch, ctx=None):
        ctx = ctx or StepContext()
        if (
            self.objective != "dspine"
            or ctx.accumulation_steps == 1
            or not torch.is_grad_enabled()
        ):
            return self._forward(batch, ctx)
        device = next(self.model.parameters()).device
        devices = [] if device.type == "cpu" else [device.index]
        device_module = torch.get_device_module(device.type)
        cpu_rng = torch.get_rng_state()
        device_rng = [device_module.get_rng_state(index) for index in devices]
        with torch.no_grad():
            output = self._forward(batch, ctx)
        replay_batch = replace(
            batch,
            tensors={key: value.detach().cpu() for key, value in batch.tensors.items()},
        )

        def replay(window_ratios):
            if self._alignment_window is not window_ratios:
                numerator, denominator = window_ratios["dspine/backbone_ce"]
                totals = torch.stack((numerator, denominator))
                if dist.is_initialized():
                    dist.all_reduce(totals)
                self._alignment_ce = (totals[0] / totals[1].clamp_min(1)).detach()
                self._alignment_window = window_ratios
            with torch.random.fork_rng(devices=devices, device_type=device.type):
                torch.set_rng_state(cpu_rng)
                for index, state in zip(devices, device_rng):
                    device_module.set_rng_state(state, index)
                return self._forward(replay_batch, ctx, self._alignment_ce)

        return replace(output, replay_loss=replay)
