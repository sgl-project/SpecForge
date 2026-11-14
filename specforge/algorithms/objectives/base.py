"""Objective contracts and the composite module executed by distributed training."""

import math
from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import Tensor, nn


class DeferredTerm(Protocol):
    """A loss term the objective hands out unfinished; the training layer finishes it."""

    def finish(self, result: "ObjectiveResult") -> "ObjectiveResult": ...


class StepStatistics(Protocol):
    """Per-forward statistics an objective observes; the training layer applies them each optimizer step."""

    @property
    def settings(
        self,
    ) -> dict[str, object]: ...  # what built them; the resume contract records it

    def observe(self, output, *, keep: Tensor) -> DeferredTerm | None: ...

    def step(self, *, process_group=None) -> dict[str, Tensor]: ...


@dataclass
class ObjectiveResult:
    numerator: Tensor
    denominator: Tensor
    ratios: dict[str, tuple[Tensor, Tensor]] = field(default_factory=dict)
    # A term the training layer finishes before the loss is read; None once finished.
    deferred: DeferredTerm | None = None

    @property
    def loss(self):
        if self.deferred is not None:
            raise ValueError(
                "a deferred term is unfinished; the training layer finishes it"
            )
        return self.numerator / self.denominator.where(self.denominator > 0, 1)


def microbatch_mean(result, *, global_denominator=False):
    """Preserve mean-of-microbatch losses; DDP averages rank gradients."""
    import torch.distributed as dist

    denominator = result.denominator.detach().clone()
    world_size = 1
    if global_denominator and dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(denominator)
    loss = world_size * result.numerator / denominator.where(denominator > 0, 1)
    return ObjectiveResult(loss, denominator.new_ones(()), result.ratios)


class ObjectiveModel(nn.Module):
    """One wrapped forward includes the draft and every differentiable loss path.

    Teacher features are immutable batch data. The frozen head only projects
    them; this module never runs a target model or owns optimizer/checkpoint I/O.
    """

    def __init__(
        self,
        draft_model,
        objective,
        target_head=None,
        *,
        registration=None,
        name=None,
        token_loss=None,
        weighting=None,
        frozen_parameters=(),
        statistics: StepStatistics | None = None,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.statistics = (
            statistics  # observed per forward, stepped by the training layer
        )
        self.objective = objective
        self.registration = registration
        self.objective_name = name
        self.target_head = target_head
        self.token_loss = token_loss
        self.weighting = weighting
        self.frozen_parameters = tuple(frozen_parameters)
        if target_head is not None:
            target_head.requires_grad_(False)
            target_head.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.target_head is not None:
            self.target_head.eval()
        return self

    def predictions(
        self,
        hidden,
        batch,
        teacher_positions,
        *,
        mask,
        labels=None,
        positions=None,
        unit_size=1,
        chain_ids=None,
        sequence_ids=None,
        projection_inputs=None,
        auxiliary_inputs=None,
        **options,
    ):
        from .inputs import gather
        from .predictions import Predictions, Teacher

        def align(value):
            return value.reshape(-1, unit_size, *value.shape[mask.ndim :])

        teacher = None
        if "target_logits" in batch:
            teacher = Teacher(
                align(gather(batch["target_logits"], teacher_positions)).detach()
            )
        elif self.target_head is not None and "target_last_hidden_states" in batch:
            teacher = Teacher(
                align(
                    gather(batch["target_last_hidden_states"], teacher_positions)
                ).detach(),
                self.target_head,
            )
        if positions is None:
            positions = torch.arange(unit_size, device=hidden.device)
        hidden = hidden.reshape(-1, unit_size, hidden.shape[-1])
        if chain_ids is None:
            chain_ids = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
        if sequence_ids is None:
            sequence_ids = torch.arange(mask.shape[0], device=hidden.device).reshape(
                -1, *([1] * (mask.ndim - 1))
            )
        return Predictions(
            hidden=hidden,
            projector=self.draft_model.project,
            mask=mask.reshape(-1, unit_size),
            chain_ids=torch.broadcast_to(chain_ids, hidden.shape[:2]),
            sequence_ids=torch.broadcast_to(sequence_ids, mask.shape).reshape(
                -1, unit_size
            ),
            labels=None if labels is None else labels.reshape(-1, unit_size),
            positions=torch.broadcast_to(positions, mask.shape).reshape(-1, unit_size),
            teacher=teacher,
            projection_inputs={
                key: align(value) for key, value in (projection_inputs or {}).items()
            },
            auxiliary_inputs={
                key: align(value) for key, value in (auxiliary_inputs or {}).items()
            },
            **options,
        )

    def deferred_term(self, output, blocks):
        """Hand this forward and its kept slots to the statistics; the deferred term, or None."""
        if self.statistics is None:
            return None
        slots = self.draft_model.config.draft.spec_config.protocol.num_model_slots
        # Every token of a kept block went through the model; padded blocks did not.
        keep = blocks.keep.repeat_interleave(slots, dim=1).reshape(-1)
        return self.statistics.observe(output, keep=keep)

    def block_predictions(self, output, blocks, batch, **options):
        protocol = self.draft_model.config.draft.spec_config.protocol
        slots = protocol.num_model_slots
        return self.predictions(
            output.hidden_states,
            batch,
            blocks.label_positions - 1,
            mask=blocks.weights,
            labels=blocks.labels,
            unit_size=slots,
            positions=torch.arange(slots, device=blocks.labels.device)
            - protocol.prediction_positions[0],
            **options,
        )

    def forward(self, batch, *, step, total_steps, **context):
        from .inputs import valid_tokens

        valid_tokens(batch)
        # Detach capture tensors centrally, including offline MTP features.
        batch = {
            key: value.detach() if isinstance(value, Tensor) else value
            for key, value in batch.items()
        }
        dtype = self.draft_model.embed_tokens.weight.dtype
        for key in ("hidden_states", "mtp_hidden_states"):
            if key in batch:
                batch[key] = batch[key].to(dtype=dtype)
        if self.target_head is not None and "target_last_hidden_states" in batch:
            # the teacher projects in its head's dtype, whatever the records hold
            batch["target_last_hidden_states"] = batch["target_last_hidden_states"].to(
                dtype=self.target_head.weight.dtype
            )
        from .execution import execute_batch

        prepared = self.objective(
            self, batch, step=step, total_steps=total_steps, **context
        )
        return execute_batch(
            prepared,
            self.token_loss,
            self.weighting,
            training=self.draft_model.training,
        )


def check_options(**groups):
    """Small scalar validators opted into by each objective, not registry policy."""
    for group, values in groups.items():
        for name, value in values.items():
            if value is None:
                continue
            valid = type(value) in (int, float) and math.isfinite(value)
            if group in ("counts", "zero_counts"):
                valid = valid and type(value) is int
            if group in ("positive", "counts"):
                valid = valid and value > 0
            else:
                valid = valid and value >= 0
            if group == "probabilities":
                valid = valid and value <= 1
            if not valid:
                raise ValueError(f"invalid {name}: {value!r} ({group})")
