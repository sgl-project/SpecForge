"""Training-only registry; no checkpoint, data-provider or launch registrations."""

from dataclasses import dataclass, replace

from .base import ObjectiveModel
from .losses import PositionWeighting, TokenLoss


@dataclass(frozen=True)
class ProtocolRegistration:
    factory: type
    architectures: tuple[str, ...]
    inputs: tuple[str, ...]
    provides: frozenset[str]
    trains: tuple[str, ...]


_PROTOCOLS = {}
_DEFAULTS = {}


def register_protocol(name, *, architectures, inputs, provides, trains, default=True):
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
        )
        if default:
            _DEFAULTS.update(dict.fromkeys(architectures, name))
        return factory

    return register


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
    architecture = draft_model.config.draft.architecture
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
    missing = (selected_loss.requires | position_weighting.requires) - entry.provides
    if missing:
        raise ValueError(
            f"protocol {name!r} lacks required capabilities: {sorted(missing)}"
        )
    entry = replace(
        entry, inputs=tuple(dict.fromkeys(entry.inputs + selected_loss.inputs))
    )
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
        token_loss=token_loss,
        weighting=position_weighting,
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
