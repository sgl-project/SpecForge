"""Training-only registry; no checkpoint, data-provider or launch registrations."""

from collections.abc import Callable
from dataclasses import dataclass, replace

from .base import ObjectiveModel
from .configuration import (
    LossConfiguration,
    loss_definitions,
    resolve_loss_configuration,
)
from .execution_options import LossExecution
from .losses import PositionWeighting, TokenLoss


@dataclass(frozen=True)
class WindowStatistic:
    """Window-wide ratio passed back to the objective during replay."""

    ratio_name: str
    argument_name: str


@dataclass(frozen=True)
class TrainingRequirements:
    """Data and execution requirements used by canonical training composition."""

    capture_method: str | None = None
    explicit_initialization: bool = False
    supports_vocab_mapping: bool = False
    required_batch_size: int | None = None
    minimum_loss_tokens: int = 1
    loss_mask_filter: Callable[[object], bool] | None = None
    attention_backends: frozenset[str] = frozenset({"eager"})
    window_statistic: WindowStatistic | None = None
    host_inputs: Callable | None = (
        None  # (batch tensors, draft config) -> extra tensors
    )


@dataclass(frozen=True)
class ProtocolRegistration:
    factory: type
    architectures: tuple[str, ...]
    inputs: tuple[str, ...]
    provides: frozenset[str]
    trains: tuple[str, ...]
    requires: frozenset[str] = frozenset()
    training: TrainingRequirements = TrainingRequirements()
    legacy_strategy: str | None = None  # Preserve existing data/rendezvous identifiers.


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
    legacy_strategy=None,
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
        legacy_strategy: Optional existing data/rendezvous strategy identifier.

    Returns:
        A decorator registering and returning the objective factory unchanged.

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
            legacy_strategy,
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
    loss: TokenLoss
    weighting: PositionWeighting
    execution: LossExecution
    losses: LossConfiguration | None = None

    @property
    def requires_teacher(self):
        return "teacher_logits" in self.registration.requires | (
            self.losses.requires if self.losses else self.loss.requires
        )


def resolve_objective(
    architecture, *, name=None, options=None, loss=None, weighting=None, execution=None
):
    """Resolve algorithm and loss settings without constructing a model.

    Args:
        architecture: Canonical draft architecture name.
        name: Optional protocol name; otherwise use the architecture's default.
        options: Algorithm-specific objective constructor options.
        loss: Common mathematical loss configuration, or algorithm defaults.
        weighting: Common position-weighting configuration, or algorithm defaults.
        execution: Loss execution options such as chunking and fused kernels.

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
    objective = entry.factory(**(options or {}))
    named = loss is not None and bool({"terms", "groups", "scheduling"} & loss.keys())
    loss_configuration = None
    if named:
        if execution:
            raise ValueError(
                "Use loss.scheduling and per-term execution instead of loss_execution"
            )
        legacy_options = set(getattr(objective, "legacy_loss_options", ())) | set(
            getattr(objective, "default_loss_options", ())
        )
        if legacy_options.intersection(options or {}):
            raise ValueError(
                "Configure loss definitions and coefficients under loss.terms/groups"
            )
        loss_configuration, execution = resolve_loss_configuration(objective, loss)
        token_loss = TokenLoss()
    else:
        token_loss = TokenLoss(**(loss or {}))
        execution = LossExecution(**(execution or {}))
    _reject_shadowed_options(
        options,
        weighting is not None,
        getattr(objective, "default_weighting_options", ()),
        "weighting",
    )
    _reject_shadowed_options(
        options,
        token_loss.kind != "default",
        getattr(objective, "default_loss_options", ()),
        "loss",
    )
    if weighting is None:
        weighting = getattr(objective, "default_weighting", None) or {}
    position_weighting = PositionWeighting(**weighting)
    selected_loss = (
        objective.default_loss if token_loss.kind == "default" else token_loss
    )
    if execution.teacher_chunk_size and not entry.training.supports_vocab_mapping:
        raise ValueError("compact teacher requires a vocabulary-aware protocol")
    missing = (
        entry.requires
        | (
            loss_configuration.requires
            if loss_configuration
            else selected_loss.requires
        )
        | position_weighting.requires
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
                + (
                    loss_configuration.inputs
                    if loss_configuration
                    else selected_loss.inputs
                )
            )
        ),
    )
    return ResolvedObjective(
        name,
        entry,
        objective,
        selected_loss,
        position_weighting,
        execution,
        loss_configuration,
    )


def build_objective_model(
    draft_model,
    *,
    name=None,
    options=None,
    frozen=(),
    target_head=None,
    loss=None,
    weighting=None,
    execution=None,
):
    """Compose a draft model with its objective and selected loss execution.

    Args:
        draft_model: Constructed canonical draft model.
        name: Optional protocol name; defaults from the draft architecture.
        options: Algorithm-specific objective options.
        frozen: Additional module/parameter paths to freeze.
        target_head: Optional teacher head; it is frozen and kept in eval mode.
        loss: Common mathematical loss configuration.
        weighting: Common position-weighting configuration.
        execution: Loss execution configuration.

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
        loss=loss,
        weighting=weighting,
        execution=execution,
    )
    name, entry, objective = resolved.name, resolved.registration, resolved.objective
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
    declared = getattr(objective, "frozen_parameters", ())
    if resolved.losses is not None:
        definitions = loss_definitions(objective)
        declared = tuple(
            module
            for key, definition in definitions.items()
            if resolved.losses.terms[key].coefficient == 0
            for module in definition.modules
        )
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
        token_loss=resolved.loss,
        loss_configuration=resolved.losses,
        weighting=resolved.weighting,
        frozen_parameters=frozen,
        execution=resolved.execution,
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
