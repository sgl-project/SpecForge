"""Training-only registry; no checkpoint, data-provider or launch registrations."""

from dataclasses import dataclass, replace

from .base import ObjectiveModel
from .losses import PositionWeighting, TokenLoss


@dataclass(frozen=True)
class ObjectiveRegistration:
    factory: type
    architectures: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    trains: tuple[str, ...]


_OBJECTIVES = {}
_DEFAULTS = {}


def register_objective(name, *, architectures, inputs, outputs, trains, default=True):
    def register(factory):
        if name in _OBJECTIVES:
            raise ValueError(f"objective already registered: {name}")
        if default and any(architecture in _DEFAULTS for architecture in architectures):
            raise ValueError("architecture already has a default objective")
        _OBJECTIVES[name] = ObjectiveRegistration(
            factory, tuple(architectures), tuple(inputs), tuple(outputs), tuple(trains)
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
    name = name or default_objective(architecture)
    if name not in _OBJECTIVES:
        raise ValueError(f"unknown objective: {name!r}")
    entry = _OBJECTIVES[name]
    if architecture not in entry.architectures:
        raise ValueError(f"objective {name!r} does not support {architecture}")
    objective = entry.factory(**(options or {}))
    token_loss = TokenLoss(**(loss or {}))
    if weighting is None:
        gamma = getattr(objective, "loss_decay_gamma", None)
        weighting = {"kind": "exponential", "gamma": gamma} if gamma is not None else {}
    position_weighting = PositionWeighting(**weighting)
    if token_loss.requires_teacher and "target_last_hidden_states" not in entry.inputs:
        entry = replace(entry, inputs=entry.inputs + ("target_last_hidden_states",))
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


def objective_registrations():
    return dict(_OBJECTIVES)


def default_objective(architecture):
    if architecture not in _DEFAULTS:
        raise ValueError(f"no default objective for {architecture!r}")
    return _DEFAULTS[architecture]
