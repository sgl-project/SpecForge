"""Training-only registry; no checkpoint, data-provider or launch registrations."""

from dataclasses import dataclass, replace

from .base import ObjectiveModel
from .losses import PositionWeighting, TokenLoss


@dataclass(frozen=True)
class TrainingRequirements:
    """Data and execution requirements used by canonical training composition."""

    capture_method: str | None = None
    strategy: str | None = None
    vocabulary: bool = False
    required_batch_size: int | None = None
    minimum_loss_tokens: int = 1
    consecutive_loss_tokens: bool = False
    dataloader_workers: int = 4
    window_statistic: tuple[str, str] | None = None  # (ratio name, objective argument)


@dataclass(frozen=True)
class ProtocolRegistration:
    factory: type
    architectures: tuple[str, ...]
    inputs: tuple[str, ...]
    provides: frozenset[str]
    trains: tuple[str, ...]
    requires: frozenset[str] = frozenset()
    training: TrainingRequirements = TrainingRequirements()


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
):
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

    @property
    def requires_teacher(self):
        return "teacher_logits" in self.registration.requires | self.loss.requires


def resolve_objective(
    architecture, *, name=None, options=None, loss=None, weighting=None
):
    """Resolve defaults and capture requirements without constructing a model."""
    name = name or default_protocol(architecture)
    if name not in _PROTOCOLS:
        raise ValueError(f"unknown objective: {name!r}")
    entry = _PROTOCOLS[name]
    if architecture not in entry.architectures:
        raise ValueError(f"objective {name!r} does not support {architecture}")
    objective = entry.factory(**(options or {}))
    token_loss = TokenLoss(**(loss or {}))
    if weighting is None:
        gamma = getattr(objective, "loss_decay_gamma", None)
        weighting = getattr(objective, "default_weighting", None)
        if weighting is None:
            weighting = (
                {"kind": "exponential", "gamma": gamma} if gamma is not None else {}
            )
    position_weighting = PositionWeighting(**weighting)
    selected_loss = (
        objective.default_loss if token_loss.kind == "default" else token_loss
    )
    missing = (
        entry.requires | selected_loss.requires | position_weighting.requires
    ) - entry.provides
    if missing:
        raise ValueError(
            f"protocol {name!r} lacks required capabilities: {sorted(missing)}"
        )
    entry = replace(
        entry, inputs=tuple(dict.fromkeys(entry.inputs + selected_loss.inputs))
    )
    return ResolvedObjective(name, entry, objective, selected_loss, position_weighting)


def build_objective_model(
    draft_model,
    *,
    name=None,
    options=None,
    frozen=(),
    target_head=None,
    loss=None,
    weighting=None,
):
    resolved = resolve_objective(
        draft_model.config.draft.architecture,
        name=name,
        options=options,
        loss=loss,
        weighting=weighting,
    )
    name, entry, objective = resolved.name, resolved.registration, resolved.objective
    frozen = tuple(frozen) + getattr(objective, "frozen_parameters", ())
    to_freeze = []
    for prefix in frozen:
        try:
            try:
                module = draft_model.get_submodule(prefix)
            except AttributeError:
                module = draft_model.get_parameter(prefix)
        except AttributeError as error:
            raise ValueError(
                f"unknown frozen parameter or module: {prefix!r}"
            ) from error
        to_freeze.append(module)
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
        weighting=resolved.weighting,
    )


def protocol_registrations():
    return dict(_PROTOCOLS)


def default_protocol(architecture):
    if architecture not in _DEFAULTS:
        raise ValueError(f"no default objective for {architecture!r}")
    return _DEFAULTS[architecture]


# Existing training callers use these names until the entry-point PR is updated.
objective_registrations = protocol_registrations
default_objective = default_protocol
register_objective = register_protocol
