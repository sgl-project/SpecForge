"""Objective contracts and the composite module executed by distributed training."""

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


@dataclass
class ObjectiveResult:
    numerator: Tensor
    denominator: Tensor
    ratios: dict[str, tuple[Tensor, Tensor]] = field(default_factory=dict)
    sums: dict[str, Tensor] = field(default_factory=dict)
    #: This rank's sums of each statistic the objective needs pooled over a
    #: window, by name; filled by a prepass only.
    statistics: dict[str, Tensor] = field(default_factory=dict)

    @property
    def loss(self):
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
    return ObjectiveResult(loss, denominator.new_ones(()), result.ratios, result.sums)


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
        loss_configuration,
        weighting=None,
        frozen_parameters=(),
        execution=None,
    ):
        super().__init__()
        from .execution_options import LossExecution
        from .implementations.registry import ExecutionSelection

        self.execution = execution or LossExecution()
        self.loss_implementations = ExecutionSelection()
        self.draft_model = draft_model
        self.objective = objective
        self.registration = registration
        self.objective_name = name
        self.target_head = target_head
        self.loss_configuration = loss_configuration
        self.weighting = weighting
        self.frozen_parameters = tuple(frozen_parameters)
        if target_head is not None:
            target_head.requires_grad_(False)
            target_head.eval()

    def adopt_recorded_execution(self, recorded):
        """Continue a resumed run with the fused setting its checkpoint recorded.

        An unset ``fused`` has no request of its own, so on resume it takes the
        recorded one: a run written with the reference formulas is not moved to
        optimized kernels by resuming it, and the other way round. An explicit
        setting is left alone and compared as it is.

        Args:
            recorded: The checkpoint's execution record, or None when it has none.

        Returns:
            The execution record this run now has.
        """
        from dataclasses import asdict, replace

        if (
            self.execution.fused is None
            and isinstance(recorded, dict)
            and recorded.get("fused") is not None
        ):
            self.execution = replace(self.execution, fused=recorded["fused"])
        return asdict(self.execution)

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
        context_inputs=None,
        projection=None,
        **options,
    ):
        """Align algorithm output and supervision into deferred prediction units.

        Args:
            hidden: Backbone hidden states, with hidden size in the last dimension.
            batch: Captured batch tensors; teacher logits or hidden states are optional.
            teacher_positions: Indices used to gather teacher supervision.
            mask: Supervision weights/eligibility aligned with the prediction positions.
            labels: Optional hard token targets aligned with mask.
            positions: Optional position/depth values used by weighting.
            unit_size: Number of positions in each indivisible projection unit.
            chain_ids: Optional acceptance-chain IDs broadcastable to the unit layout.
            sequence_ids: Optional source-sequence IDs broadcastable to mask.
            projection_inputs: Extra aligned tensors passed to the projection.
            context_inputs: Extra aligned tensors consumed by the chunk computation.
            projection: Optional adapter; defaults to the draft's ordinary project.
            **options: Remaining Predictions fields, such as plan and denominator.

        Returns:
            Predictions with tensors aligned to [units, positions, ...]. Draft
            projection is deferred; teacher data is gathered but not projected.
        """
        from .inputs import gather
        from .predictions import Predictions, Teacher
        from .projection import Projection

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
        from .plan import LossPlan

        options["plan"] = self.loss_configuration.bind(options.get("plan", LossPlan()))
        return Predictions(
            hidden=hidden,
            projection=projection or Projection(self.draft_model.project),
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
            context_inputs={
                key: align(value) for key, value in (context_inputs or {}).items()
            },
            **options,
        )

    def block_predictions(self, output, blocks, batch, **options):
        """Build deferred predictions for complete algorithm proposal blocks.

        Args:
            output: Draft forward output containing hidden_states.
            blocks: Prepared tokens, labels, masks and teacher positions.
            batch: Captured batch inputs used to gather teacher data.
            **options: Projection, loss plan and other prediction overrides.

        Returns:
            Predictions preserving the configured number of model slots per block.
        """
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

    def host_inputs(self, tensors):
        """Extra batch tensors the objective derives on the host, before placement."""
        prepare = self.registration.training.host_inputs if self.registration else None
        if prepare is None:
            return {}
        return prepare(tensors, self.draft_model.config.draft)

    def kernel_record(self):
        """The loss kernels resolved so far, by the group they compute."""
        return dict(self.loss_implementations.names)

    def draft_state(self, state):
        """The draft's entries of this module's state dict, under the draft's keys.

        A parameter the draft holds under several names is kept under its
        first name only.
        """
        seen, repeated = set(), set()
        for key, value in self.draft_model.named_parameters(remove_duplicate=False):
            if id(value) in seen:
                repeated.add(key)
            seen.add(id(value))
        prefix = "draft_model."
        return {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix) and key[len(prefix) :] not in repeated
        }

    def replay_arguments(self, replay_values, *, prepass=False):
        """The forward's keyword argument that carries the declared window statistic.

        Zero in a prepass. In a replay, the ratio of the statistic's sums over
        the whole optimizer window and every rank, guarded against an empty
        denominator and detached. Otherwise None: the objective then uses the
        batch's own value. Empty for an objective without a statistic.

        A runner calls this outside its wrapper and passes the result through
        it, so the division is done in the precision the sums were pooled in
        and a wrapper that casts the forward's inputs casts the value.

        Args:
            replay_values: The pooled sums by statistic name; empty outside a
                replay.
            prepass: Whether the forward only collects the statistic.
        """
        statistic = (
            self.registration.training.window_statistic if self.registration else None
        )
        if statistic is None:
            return {}
        if prepass:
            value = torch.zeros((), device=next(self.parameters()).device)
        elif replay_values:
            numerator, denominator = replay_values[statistic.argument_name]
            denominator = denominator.where(denominator > 0, 1)
            value = (numerator / denominator).detach()
        else:
            value = None
        return {statistic.argument_name: value}

    def forward(self, batch, *, step, total_steps, prepass=False, **context):
        """Run the objective on one batch.

        Args:
            batch: Captured batch tensors.
            step: Optimizer updates applied so far.
            total_steps: The schedule's horizon, or None.
            prepass: Collect the window statistic only: the result carries
                this rank's sums of it.
            **context: Passed to the objective, with what ``replay_arguments``
                returned for this call.
        """
        from .inputs import valid_tokens

        statistic = (
            self.registration.training.window_statistic if self.registration else None
        )
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
        result = execute_batch(
            prepared,
            None,
            self.weighting,
            training=self.draft_model.training,
            execution=self.execution,
            selection=self.loss_implementations,
        )
        if prepared.diagnostics is not None:
            with torch.no_grad():
                ratios, sums = prepared.diagnostics()
            result.ratios.update(ratios)
            result.sums.update(sums)
        if statistic is not None and prepass:
            result.statistics[statistic.argument_name] = torch.stack(
                result.ratios[statistic.ratio_name]
            )
        return result


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
