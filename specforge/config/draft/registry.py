"""Configuration schemas and pure validation; computational builders live in modeling."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from inspect import isabstract
from typing import TYPE_CHECKING, Callable

from .architecture import (
    ArchitectureSpec,
    DFlash2Spec,
    DFlashSpec,
    DSparkSpec,
    DSpineSpec,
)
from .architecture.constraints import (
    validate_dflash,
    validate_dflash2,
    validate_target_context,
)
from .architecture.dspine import validate_dspine
from .attention import AttentionSpec, GQASpec, MHASpec, MLASpec
from .conv import ConvSpec, GroupedCausalConvSpec
from .conv.constraints import validate_grouped_causal
from .logits import MuPLogitTransformSpec, NoLogitTransformSpec, TargetLogitSpec
from .mlp import DenseFFNSpec, FFNSpec
from .norm import NormSpec, RMSNormSpec
from .residual import ResidualSpec, StandardResidualSpec
from .rope import DefaultRoPESpec, LinearRoPESpec, NoRoPESpec, RopeSpec

if TYPE_CHECKING:
    from .model import DraftConfig

_BASES = {
    "attention": AttentionSpec,
    "mlp": FFNSpec,
    "conv": ConvSpec,
    "norm": NormSpec,
    "residual": ResidualSpec,
    "rope": RopeSpec,
    "logits": TargetLogitSpec,
}
_SCHEMAS: dict[tuple[str, str], type] = {}
_VALIDATORS: dict[tuple[str, str], Callable[..., None]] = {}


def _validate_schema(schema: type, label: str) -> None:
    if (
        "__dataclass_fields__" not in schema.__dict__
        or not schema.__dataclass_params__.frozen
        or isabstract(schema)
    ):
        raise TypeError(f"{label} schema must be a concrete frozen dataclass")
    for field in fields(schema):
        if field.name in ("type", "rope_type"):
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


def component_type(kind: str, node):
    """Read the discriminator, preserving RoPE's serialized rope_type name."""
    return getattr(node, "rope_type" if kind == "rope" else "type", None)


def register_component_schema(
    kind: str, schema: type, *, validate: Callable[..., None] | None = None
) -> None:
    """Register a frozen node under its kind and class-level type name.

    Repeating the same registration is harmless; replacing a schema is an
    error. LayerSpec is a fixed reference schema and is not registered here.
    An optional pure validator accepts the node and caller-supplied context.
    Omitting it on a repeated registration preserves the existing validator.
    """
    if not isinstance(kind, str) or kind not in _BASES:
        raise ValueError(f"unknown component kind {kind!r}; expected {sorted(_BASES)}")
    if not isinstance(schema, type) or not issubclass(schema, _BASES[kind]):
        raise TypeError(f"{kind} schema must inherit {_BASES[kind].__name__}")
    _validate_schema(schema, kind)
    if validate is not None and not callable(validate):
        raise TypeError("component validation must be callable")
    name = component_type(kind, schema)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{kind} schema must declare a nonempty type name")
    key = (kind, name)
    existing = _SCHEMAS.get(key)
    if existing is not None and existing is not schema:
        raise ValueError(
            f"component {kind}/{name} already registered to {existing.__name__}"
        )
    previous = _VALIDATORS.get(key)
    if validate is not None and previous is not None and previous is not validate:
        raise ValueError(f"component validation {kind}/{name} already registered")
    _SCHEMAS[key] = schema
    if validate is not None:
        _VALIDATORS[key] = validate


def component_schema(kind: str, name: str) -> type:
    """Resolve a schema without importing a model, backend, or training code."""
    if not isinstance(kind, str) or kind not in _BASES:
        raise ValueError(f"unknown component kind {kind!r}")
    if not isinstance(name, str) or (kind, name) not in _SCHEMAS:
        available = sorted(key[1] for key in _SCHEMAS if key[0] == kind)
        raise ValueError(f"unknown {kind} type {name!r}; available: {available}")
    return _SCHEMAS[kind, name]


def validate_component(kind: str, node, **context) -> None:
    """Run the registered context validator; node-local checks run at construction."""
    name = component_type(kind, node)
    if type(node) is not component_schema(kind, name):
        raise ValueError(f"{kind}/{name}: expected its registered configuration node")
    validator = _VALIDATORS.get((kind, name))
    if validator is not None:
        validator(node, **context)


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
    ("norm", RMSNormSpec),
    ("residual", StandardResidualSpec),
    ("rope", DefaultRoPESpec),
    ("rope", LinearRoPESpec),
    ("rope", NoRoPESpec),
    ("logits", NoLogitTransformSpec),
    ("logits", MuPLogitTransformSpec),
):
    register_component_schema(_kind, _schema)

register_component_schema(
    "conv", GroupedCausalConvSpec, validate=validate_grouped_causal
)

register_spec("DFlashDraftModel", DFlashSpec, validate_dflash)
register_spec("DFlash2DraftModel", DFlash2Spec, validate_dflash2)
register_spec("DSparkDraftModel", DSparkSpec, validate_target_context)
register_spec("DSpineDraftModel", DSpineSpec, validate_dspine)
