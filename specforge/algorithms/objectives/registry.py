"""Training-only registry; no checkpoint, data-provider or launch registrations."""

from dataclasses import dataclass

from .base import ObjectiveModel


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
    draft_model, *, name=None, options=None, frozen=(), target_head=None
):
    architecture = draft_model.config.draft.architecture
    name = name or default_objective(architecture)
    if name not in _OBJECTIVES:
        raise ValueError(f"unknown objective: {name!r}")
    entry = _OBJECTIVES[name]
    if architecture not in entry.architectures:
        raise ValueError(f"objective {name!r} does not support {architecture}")
    objective = entry.factory(**(options or {}))
    for prefix in frozen:
        try:
            module = draft_model.get_submodule(prefix)
        except AttributeError:
            module = draft_model.get_parameter(prefix)
        module.requires_grad_(False)
    uncovered = [
        key
        for key, parameter in draft_model.named_parameters()
        if parameter.requires_grad
        and not any(
            key == prefix or key.startswith(prefix + ".") for prefix in entry.trains
        )
    ]
    if uncovered:
        raise ValueError(
            f"objective {name!r} does not train serving parameters: {uncovered}"
        )
    return ObjectiveModel(
        draft_model, objective, target_head, registration=entry, name=name
    )


def objective_registrations():
    return dict(_OBJECTIVES)


def default_objective(architecture):
    if architecture not in _DEFAULTS:
        raise ValueError(f"no default objective for {architecture!r}")
    return _DEFAULTS[architecture]
