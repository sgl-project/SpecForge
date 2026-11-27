"""Training-only registry; no checkpoint, data-provider or launch registrations."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Optional

from .base import ObjectiveModel, SequenceParallelGroups, validate_anchor_sampling
from .configuration import (
    LossConfiguration,
    loss_definitions,
    resolve_loss_configuration,
)
from .execution_options import LossExecution
from .losses import PositionWeighting


@dataclass(frozen=True)
class WindowStatistic:
    """Window-wide ratio passed back to the objective during replay."""

    ratio_name: str
    argument_name: str

    def __post_init__(self):
        # The objective model's forward and the step runner pass these keywords
        # explicitly alongside ``**replay``.
        if self.argument_name in (
            "batch",
            "step",
            "total_steps",
            "prepass",
            "collect_detailed_metrics",
        ):
            raise ValueError(
                f"a window statistic cannot be passed as {self.argument_name!r}: "
                "the objective model's forward takes that keyword itself"
            )


@dataclass(frozen=True)
class TrainingRequirements:
    """Data and execution requirements used by canonical training composition."""

    capture_method: str | None = None
    explicit_initialization: bool = False
    supports_vocab_mapping: bool = False
    required_batch_size: int | None = None
    minimum_loss_tokens: int = 1
    loss_mask_filter: Callable[[object], bool] | None = None
    supports_sequence_parallel: bool = False
    attention_backends: frozenset[str] = frozenset({"eager", "sdpa", "flex_attention"})
    window_statistic: WindowStatistic | None = None
    host_inputs: Callable | None = (
        None  # (batch tensors, draft config) -> extra tensors
    )
    #: Feature-loader workers when data.dataloader_num_workers is unset; None
    #: leaves the trainer's own default.
    dataloader_workers: int | None = None
    #: Whether an unset model.use_liger_kernel enables Liger RMSNorm/SwiGLU on a
    #: CUDA/ROCm trainer with a silu/swish draft. Only set where that
    #: combination was validated.
    liger_by_default: bool = False
    #: Whether evaluation runs the objective's diagnostics, for an objective
    #: whose evaluation reports them.
    evaluation_diagnostics: bool = False
    #: How a runner normalizes the loss: "window" pools one denominator over
    #: the optimizer window and the ranks; "microbatch" pools every term's
    #: denominator over the ranks of each micro-batch (the protocol's batches
    #: use the "global_microbatch" reduction).
    reduction: str = "window"
    #: Whether the objective samples block anchors (``prepare_blocks``); only
    #: such an objective can have its anchors drawn on the host.
    samples_anchors: bool = False
    #: Uniform numbers the objective draws per anchor besides the anchors
    #: (DSpine's replacement draw); a host draw provides them with the anchors.
    draws_per_anchor: int = 0

    def __post_init__(self):
        if self.reduction not in ("window", "microbatch"):
            raise ValueError(f"unknown reduction: {self.reduction!r}")
        if self.draws_per_anchor and not self.samples_anchors:
            raise ValueError("draws_per_anchor needs an objective that samples anchors")


@dataclass(frozen=True)
class ProtocolRegistration:
    factory: type
    architectures: tuple[str, ...]
    inputs: tuple[str, ...]
    provides: frozenset[str]
    trains: tuple[str, ...]
    requires: frozenset[str] = frozenset()
    training: TrainingRequirements = TrainingRequirements()
    runtime_name: str | None = None  # Preserve existing data/rendezvous identifiers.
    resolve_requires: Callable | None = None


_PROTOCOLS = {}
_DEFAULTS = {}


def register_protocol(
    name,
    *,
    architectures,
    inputs,
    provides,
    trains,
    requires=(),
    default=True,
    training=TrainingRequirements(),
    runtime_name=None,
    resolve_requires=None,
):
    """Register an algorithm objective and its training requirements.

    Args:
        name: Unique protocol identifier.
        architectures: Canonical architecture names supported by the protocol.
        inputs: Required batch tensor names.
        provides: Capabilities made available by the protocol's predictions.
        trains: Module paths that the objective trains.
        requires: Capabilities required regardless of the selected common loss.
        default: Whether this becomes the default for the listed architectures.
        training: Data, capture and execution requirements used by composition.
        runtime_name: Stable data/checkpoint identifier, when different from the objective.

    Returns:
        A decorator registering and returning the objective factory unchanged.

    The optional resolve_requires callback receives the configured objective
    and returns extra input capabilities, such as teacher logits for diagnostics.

    Raises:
        ValueError: The name or an architecture's default is already registered.
    """

    def register(factory):
        if name in _PROTOCOLS:
            raise ValueError(f"objective already registered: {name}")
        if default and any(architecture in _DEFAULTS for architecture in architectures):
            raise ValueError("architecture already has a default objective")
        _PROTOCOLS[name] = ProtocolRegistration(
            factory,
            tuple(architectures),
            tuple(inputs),
            frozenset(provides),
            tuple(trains),
            frozenset(requires),
            training,
            runtime_name,
            resolve_requires,
        )
        if default:
            _DEFAULTS.update(dict.fromkeys(architectures, name))
        return factory

    return register


@dataclass(frozen=True)
class ResolvedObjective:
    name: str
    registration: ProtocolRegistration
    objective: object
    weighting: PositionWeighting
    execution: LossExecution
    losses: LossConfiguration

    @property
    def frozen_parameters(self):
        """Optional modules disabled by zero loss coefficients."""
        return tuple(
            module
            for key, definition in loss_definitions(self.objective).items()
            if self.losses.terms[key].coefficient == 0
            for module in definition.modules
        )

    @property
    def requires_teacher(self):
        return "teacher_logits" in self.registration.requires | self.losses.requires


def _training_options(factory, options, training):
    """Let a protocol inherit defaults from the run before constructing it."""
    resolve = getattr(factory, "resolve_training_options", None)
    return resolve(options, dict(training)) if resolve is not None else options


def objective_record(name, options, *, training=None):
    """Options that fix the training trajectory: what a resume compares.

    An objective lists the options that only select what is logged in
    ``logging_options``; they are left out, for a saved record as well.
    """
    factory = _PROTOCOLS[name].factory
    if training is not None:
        options = _training_options(factory, options, training)
    logged = getattr(factory, "logging_options", ())
    return {key: value for key, value in options.items() if key not in logged}


def resolve_objective(
    architecture, *, name=None, options=None, loss=None, weighting=None, training=None
):
    """Resolve algorithm and loss settings without constructing a model.

    Args:
        architecture: Canonical draft architecture name.
        name: Optional protocol name; otherwise use the architecture's default.
        options: Algorithm-specific objective constructor options.
        training: Optional run settings used by the protocol to fill omitted options.
        loss: Common mathematical loss configuration, or algorithm defaults.
        weighting: Common position-weighting configuration, or algorithm defaults.

    Returns:
        ResolvedObjective with the objective, concrete loss/weighting, execution
        options and registration updated with required capture inputs.

    Raises:
        ValueError: Selection is unknown/incompatible, options are shadowed,
            or required capabilities are unavailable.
    """
    name = name or default_protocol(architecture)
    if name not in _PROTOCOLS:
        raise ValueError(f"unknown objective: {name!r}")
    entry = _PROTOCOLS[name]
    if architecture not in entry.architectures:
        raise ValueError(f"objective {name!r} does not support {architecture}")
    options = dict(options or {})
    if training is not None:
        options = _training_options(entry.factory, options, training)
    objective = entry.factory(**options)
    if entry.resolve_requires is not None:
        entry = replace(
            entry, requires=entry.requires | entry.resolve_requires(objective)
        )
    loss_configuration, execution = resolve_loss_configuration(
        objective, {} if loss is None else loss
    )
    _reject_shadowed_options(
        options,
        weighting is not None,
        getattr(objective, "default_weighting_options", ()),
        "weighting",
    )
    if weighting is None:
        weighting = getattr(objective, "default_weighting", None) or {}
    position_weighting = PositionWeighting(**weighting)
    if execution.teacher_chunk_size and not entry.training.supports_vocab_mapping:
        raise ValueError("compact teacher requires a vocabulary-aware protocol")
    missing = (
        entry.requires | loss_configuration.requires | position_weighting.requires
    ) - entry.provides
    if missing:
        raise ValueError(
            f"protocol {name!r} lacks required capabilities: {sorted(missing)}"
        )
    entry = replace(
        entry,
        inputs=tuple(
            dict.fromkeys(
                entry.inputs
                + loss_configuration.inputs
                + (
                    ("target_last_hidden_states",)
                    if "teacher_logits" in entry.requires
                    else ()
                )
            )
        ),
    )
    return ResolvedObjective(
        name,
        entry,
        objective,
        position_weighting,
        execution,
        loss_configuration,
    )


def refuse_dpace_over_sequence_shards(weighting, *, sp_size):
    """Refuse D-PACE weighting for a sequence that is split over ranks.

    Args:
        weighting: The resolved position weighting.
        sp_size: Number of ranks one sequence is split over.

    Raises:
        ValueError: The weighting is D-PACE and ``sp_size`` is above one.
    """
    from specforge.losses.position_weights import DPACE_LOSS_TYPES

    if sp_size > 1 and weighting.kind in DPACE_LOSS_TYPES:
        raise ValueError(
            "D-PACE weighting is not supported with sequence parallelism: "
            "chain weights are normalized within each sequence shard"
        )


def build_objective_model(
    draft_model,
    *,
    name=None,
    options=None,
    training=None,
    frozen=(),
    target_head=None,
    loss=None,
    weighting=None,
    frozen_modules=(),
    attention_backend=None,
    sequence_parallel: Optional[SequenceParallelGroups] = None,
    anchor_sampling="device",
    anchor_seed=None,
):
    """Compose a draft model with its objective and selected loss execution.

    Args:
        draft_model: Constructed canonical draft model.
        name: Optional protocol name; defaults from the draft architecture.
        options: Algorithm-specific objective options.
        training: Optional run settings used by the protocol to fill omitted options.
        frozen: Additional module/parameter paths to freeze.
        target_head: Optional teacher head; it is frozen and kept in eval mode.
        loss: Common mathematical loss configuration.
        weighting: Common position-weighting configuration.
        frozen_modules: Frozen modules of the composed model that a runner
            keeps out of sharding and out of the optimizer (bound target
            tables, the teacher head).
        attention_backend: The attention backend the run trains with, for a
            protocol whose rollout depends on it.
        sequence_parallel: The process groups of the sequence-parallel axis
            (the ring group, the Ulysses group and their product, over which
            loss statistics of one sequence are reduced). Given with the
            ``usp`` backend and with no other.
        anchor_sampling: "device" draws block anchors in the forward from the
            global generator; "host" draws them before placement from
            (anchor_seed, sample id, epoch). Only an objective that samples
            anchors accepts "host".
        anchor_seed: The run's seed for host-drawn anchors.

    Returns:
        ObjectiveModel wrapping the existing draft and optional teacher head.
        Parameter freezing is applied to these modules after compatibility checks.

    Raises:
        ValueError: Requested capabilities, vocabulary optimization, or freezing
            settings are incompatible with the model/protocol.
    """
    resolved = resolve_objective(
        draft_model.config.draft.architecture,
        name=name,
        options=options,
        training=training,
        loss=loss,
        weighting=weighting,
    )
    name, entry, objective = resolved.name, resolved.registration, resolved.objective
    # Before anything below changes the supplied draft (freezing).
    validate_anchor_sampling(name, entry.training, anchor_sampling, anchor_seed)
    if (
        attention_backend is not None
        and attention_backend not in entry.training.attention_backends
    ):
        raise ValueError(
            f"objective {name!r} does not train with attention backend "
            f"{attention_backend!r}; it supports "
            f"{sorted(entry.training.attention_backends)}"
        )
    if sequence_parallel is not None and not entry.training.supports_sequence_parallel:
        raise ValueError(f"objective {name!r} does not support sequence parallelism")
    if attention_backend == "usp" and sequence_parallel is None:
        raise ValueError("attention backend 'usp' needs the sequence-parallel groups")
    if sequence_parallel is not None:
        if attention_backend != "usp":
            raise ValueError(
                "sequence-parallel groups need attention backend 'usp', not "
                f"{attention_backend!r}: no other backend splits the sequence, "
                "and its loss statistics would still be reduced over the groups"
            )
        refuse_dpace_over_sequence_shards(
            resolved.weighting, sp_size=sequence_parallel.size
        )
    if resolved.execution.teacher_chunk_size:
        from torch import nn

        from specforge.modeling.draft.bindings import BoundLMHead

        if not (
            isinstance(target_head, BoundLMHead)
            or isinstance(target_head, nn.Linear)
            and target_head.bias is None
        ):
            raise ValueError("compact teacher requires a linear teacher head")
        draft = draft_model.config.draft
        if draft.draft_vocab_size >= draft.vocab_size:
            raise ValueError("compact teacher requires a reduced draft vocabulary")
    requested = tuple(frozen)
    # a protocol may freeze an optional head this draft disables (set to None)
    declared = resolved.frozen_parameters
    frozen, to_freeze = [], []
    for prefix in requested + tuple(declared):
        try:
            try:
                module = draft_model.get_submodule(prefix)
            except AttributeError:
                module = draft_model.get_parameter(prefix)
        except AttributeError as error:
            if (
                prefix in declared
                and prefix not in requested
                and hasattr(draft_model, prefix)
                and getattr(draft_model, prefix) is None
            ):
                continue
            raise ValueError(
                f"unknown frozen parameter or module: {prefix!r}"
            ) from error
        frozen.append(prefix)
        to_freeze.append(module)
    frozen = tuple(frozen)
    uncovered = [
        key
        for key, parameter in draft_model.named_parameters()
        if parameter.requires_grad
        and not any(
            key == prefix or key.startswith(prefix + ".")
            for prefix in entry.trains + frozen
        )
    ]
    if uncovered:
        raise ValueError(
            f"objective {name!r} does not train serving parameters: {uncovered}"
        )
    for module in to_freeze:
        module.requires_grad_(False)
    return ObjectiveModel(
        draft_model,
        objective,
        target_head,
        registration=entry,
        name=name,
        loss_configuration=resolved.losses,
        weighting=resolved.weighting,
        frozen_parameters=frozen,
        execution=resolved.execution,
        frozen_modules=frozen_modules,
        attention_backend=attention_backend,
        sequence_parallel=sequence_parallel,
        anchor_sampling=anchor_sampling,
        anchor_seed=anchor_seed,
    )


def _reject_shadowed_options(options, replaced, names, selection):
    given = {key for key, value in (options or {}).items() if value is not None}
    shadowed = sorted(given & set(names)) if replaced else []
    if shadowed:
        raise ValueError(
            f"{shadowed} only configure the default {selection}; "
            f"set them in the explicit {selection} instead"
        )


def protocol_registrations():
    """Return a shallow copy of the protocol-name to registration mapping."""
    return dict(_PROTOCOLS)


def default_protocol(architecture):
    """Return the default protocol name for an architecture.

    Args:
        architecture: Canonical draft architecture name.

    Returns:
        The registered default protocol identifier.

    Raises:
        ValueError: No default is registered for the architecture.
    """
    if architecture not in _DEFAULTS:
        raise ValueError(f"no default objective for {architecture!r}")
    return _DEFAULTS[architecture]


# Existing training callers use these names until the entry-point PR is updated.
objective_registrations = protocol_registrations
default_objective = default_protocol
register_objective = register_protocol
