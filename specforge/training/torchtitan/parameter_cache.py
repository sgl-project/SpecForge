"""Reuse SimpleFSDP parameter materializations within one joint model call.

SimpleFSDP exposes parameters through its own properties, so PyTorch's
``parametrize.cached()`` does not apply. Re-reading a property otherwise creates
independent BF16 casts whose gradients accumulate in FP32. Native FSDP2 instead
accumulates uses of the same unsharded BF16 parameter before the reduction cast.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch
from torch import nn
from torchtitan.experiments.graph_trainer import simple_fsdp

_MEMBERS = "_specforge_graph_parameter_cache_members"
_wrapper_classes: dict[type, type] = {}


@dataclass
class _Cache:
    owner: nn.Module
    members: frozenset[int]
    values: dict[tuple, torch.Tensor] = field(default_factory=dict)


_active: ContextVar[tuple[_Cache, ...]] = ContextVar(
    "specforge_graph_parameter_caches", default=()
)


def _cached_property(name: str, original: property) -> property:
    def get(module):
        # Initialization and DCP inspection can explicitly request the original
        # sharded parameters, including while an outer joint scope is active.
        if simple_fsdp._active_parametrization:
            for cache in reversed(_active.get()):
                if id(module) not in cache.members:
                    continue
                # functional_call may temporarily substitute a different leaf.
                # A no-grad inspection must not poison the subsequent gradient
                # graph when both happen inside an enclosing training scope.
                key = (
                    id(module),
                    name,
                    id(module._parameters[name]),
                    torch.is_grad_enabled(),
                )
                if key not in cache.values:
                    cache.values[key] = original.__get__(module, type(module))
                return cache.values[key]
        return original.__get__(module, type(module))

    return property(get, doc=original.__doc__)


def install_graph_parameter_cache(model: nn.Module) -> None:
    """Wrap this model's generated SimpleFSDP properties without changing leaves.

    Call after ``apply_simple_fsdp``. The reusable wrapper classes affect only
    instances explicitly installed here; upstream classes and parametrization
    implementations remain untouched. Parameter names and state dicts are stable.
    """
    members = set()
    for module in model.modules():
        cls = type(module)
        if cls in _wrapper_classes.values():
            members.add(id(module))
            continue
        if not cls.__name__.startswith("SimpleFSDP"):
            continue
        properties = {
            name: value
            for name, value in vars(cls).items()
            if name in module._parameters and isinstance(value, property)
        }
        if not properties:
            continue
        wrapper = _wrapper_classes.get(cls)
        if wrapper is None:
            class_name = f"SpecForgeCached{cls.__name__}_{len(_wrapper_classes)}"
            wrapper = type(
                class_name,
                (cls,),
                {
                    "__module__": __name__,
                    **{
                        name: _cached_property(name, prop)
                        for name, prop in properties.items()
                    },
                },
            )
            globals()[class_name] = wrapper
            _wrapper_classes[cls] = wrapper
        module.__class__ = wrapper
        members.add(id(module))
    object.__setattr__(model, _MEMBERS, frozenset(members))


@contextmanager
def graph_parameter_cache(model: nn.Module) -> Iterator[None]:
    """Keep one materialization per parameter read for this joint execution.

    Same-owner nested calls reuse the enclosing cache. The outer trainer scope
    includes backward recomputation; a model-forward scope also handles direct
    validation. Every outer exit clears tensor references, including exceptions.
    An uninstalled model is a no-op to preserve ordinary trainer test fixtures.
    """
    members = getattr(model, _MEMBERS, ())
    if not members:
        yield
        return
    active = _active.get()
    if any(cache.owner is model for cache in active):
        yield
        return
    cache = _Cache(model, members)
    token = _active.set((*active, cache))
    try:
        yield
    finally:
        cache.values.clear()
        _active.reset(token)
