"""Configuration schemas only; computational builders register in modeling."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from inspect import isabstract
from typing import TYPE_CHECKING, Callable

from .architecture import (
    ArchitectureSpec,
    DFlash2Spec,
    DFlashSpec,
    DominoSpec,
    DSparkSpec,
    DSpineSpec,
    Eagle3Spec,
    PEagleSpec,
)
from .architecture.constraints import (
    validate_dflash,
    validate_dflash2,
    validate_target_context,
)
from .architecture.dspine import validate_dspine
from .architecture.eagle3 import validate_eagle3
from .architecture.eagle_fusion import validate_auxiliary_fusion
from .attention import AttentionSpec, GQASpec, MHASpec, MLASpec
from .conv import ConvSpec, GroupedCausalConvSpec
from .mlp import DenseFFNSpec, FFNSpec
from .norm import NormSpec, RMSNormSpec
from .residual import ResidualSpec, StandardResidualSpec

if TYPE_CHECKING:
    from .model import DraftConfig

_BASES = {
    "attention": AttentionSpec,
    "mlp": FFNSpec,
    "conv": ConvSpec,
    "norm": NormSpec,
    "residual": ResidualSpec,
}
_SCHEMAS: dict[tuple[str, str], type] = {}


def _validate_schema(schema: type, label: str) -> None:
    if (
        "__dataclass_fields__" not in schema.__dict__
        or not schema.__dataclass_params__.frozen
        or isabstract(schema)
    ):
        raise TypeError(f"{label} schema must be a concrete frozen dataclass")
    for field in fields(schema):
        if field.name == "type":
            raise TypeError(
                "component type must be a ClassVar, not a configuration field"
            )
        if field.init and (
            field.default is not MISSING or field.default_factory is not MISSING
        ):
            raise TypeError(
                f"{label}.{field.name}: structural fields must have no defaults"
            )
    if issubclass(schema, (AttentionSpec, ConvSpec, ResidualSpec)):
        if type(schema.capabilities) is not frozenset or any(
            type(name) is not str or not name for name in schema.capabilities
        ):
            raise TypeError(f"{label} capabilities must be a frozenset of names")


def register_component_schema(kind: str, schema: type) -> None:
    """Register a frozen node under its kind and class-level type name.

    Repeating the same registration is harmless; replacing a schema is an
    error. LayerSpec is a fixed reference schema and is not registered here.
    """
    if not isinstance(kind, str) or kind not in _BASES:
        raise ValueError(f"unknown component kind {kind!r}; expected {sorted(_BASES)}")
    if not isinstance(schema, type) or not issubclass(schema, _BASES[kind]):
        raise TypeError(f"{kind} schema must inherit {_BASES[kind].__name__}")
    _validate_schema(schema, kind)
    name = getattr(schema, "type", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{kind} schema must declare a nonempty type name")
    key = (kind, name)
    existing = _SCHEMAS.get(key)
    if existing is not None and existing is not schema:
        raise ValueError(
            f"component {kind}/{name} already registered to {existing.__name__}"
        )
    _SCHEMAS[key] = schema


def component_schema(kind: str, name: str) -> type:
    """Resolve a schema without importing a model, backend, or training code."""
    if not isinstance(kind, str) or kind not in _BASES:
        raise ValueError(f"unknown component kind {kind!r}")
    if not isinstance(name, str) or (kind, name) not in _SCHEMAS:
        available = sorted(key[1] for key in _SCHEMAS if key[0] == kind)
        raise ValueError(f"unknown {kind} type {name!r}; available: {available}")
    return _SCHEMAS[kind, name]


@dataclass(frozen=True)
class SpecRegistration:
    schema: type[ArchitectureSpec]
    validate: Callable[[DraftConfig], None]


_SPECS: dict[str, SpecRegistration] = {}


def register_spec(
    name: str, schema: type[ArchitectureSpec], validate: Callable[[DraftConfig], None]
) -> None:
    """Register an architecture schema and its pure cross-node constraints."""
    if type(name) is not str or not name.isidentifier():
        raise ValueError("architecture name must be a class identifier")
    if not isinstance(schema, type) or not issubclass(schema, ArchitectureSpec):
        raise TypeError("architecture schema must inherit ArchitectureSpec")
    _validate_schema(schema, name)
    if not callable(validate):
        raise TypeError("architecture validation must be callable")
    registration = SpecRegistration(schema, validate)
    existing = _SPECS.get(name)
    if existing is not None and existing != registration:
        raise ValueError(f"architecture {name!r} already registered")
    _SPECS[name] = registration


def architecture_spec(name: str) -> SpecRegistration:
    if type(name) is not str or name not in _SPECS:
        raise ValueError(f"unknown architecture {name!r}; available: {sorted(_SPECS)}")
    return _SPECS[name]


for _kind, _schema in (
    ("attention", GQASpec),
    ("attention", MHASpec),
    ("attention", MLASpec),
    ("mlp", DenseFFNSpec),
    ("conv", GroupedCausalConvSpec),
    ("norm", RMSNormSpec),
    ("residual", StandardResidualSpec),
):
    register_component_schema(_kind, _schema)

register_spec("DFlashDraftModel", DFlashSpec, validate_dflash)
register_spec("DFlash2DraftModel", DFlash2Spec, validate_dflash2)
register_spec("DSparkDraftModel", DSparkSpec, validate_target_context)
register_spec("DSpineDraftModel", DSpineSpec, validate_dspine)
register_spec("DominoDraftModel", DominoSpec, validate_target_context)
register_spec("Eagle3DraftModel", Eagle3Spec, validate_eagle3)
register_spec("PEagleDraftModel", PEagleSpec, validate_auxiliary_fusion)
