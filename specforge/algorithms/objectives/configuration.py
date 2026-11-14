"""Named mathematical losses and a shared optimization policy."""

import math
from dataclasses import asdict, dataclass, replace

from .execution_options import LossExecution
from .losses import TokenLoss


@dataclass(frozen=True)
class LossDefinition:
    """Algorithm-owned input contract and default mathematical definition."""

    loss: TokenLoss
    coefficient: float = 1.0
    modules: tuple[str, ...] = ()
    capabilities: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ConfiguredLoss:
    loss: TokenLoss
    coefficient: float


@dataclass(frozen=True)
class LossConfiguration:
    """Resolved formulas and the shared fusion switch; no model tensors."""

    terms: dict[str, ConfiguredLoss]
    fused: bool = False

    @property
    def requires(self):
        return frozenset().union(*(term.loss.requires for term in self.terms.values()))

    @property
    def inputs(self):
        return tuple(
            dict.fromkeys(
                name for term in self.terms.values() for name in term.loss.inputs
            )
        )

    def bind(self, plan):
        """Bind an algorithm plan to validated settings, preserving schedule factors."""
        names = tuple(
            term.name or ("token" if term.output == "logits" else term.output)
            for term in plan.terms
        )
        if len(set(names)) != len(names) or set(names) != set(self.terms):
            raise ValueError(
                f"Algorithm loss terms {names} do not match configured terms {tuple(self.terms)}"
            )
        terms = tuple(
            replace(
                term,
                name=name,
                loss=self.terms[name].loss,
                coefficient=self.terms[name].coefficient * term.coefficient,
            )
            for name, term in zip(names, plan.terms)
        )
        return replace(plan, terms=terms)

    def to_dict(self):
        """Stable effective settings persisted in the resume contract."""
        return {
            "fused": self.fused,
            "terms": {
                name: {**asdict(term.loss), "coefficient": term.coefficient}
                for name, term in self.terms.items()
            },
        }


def loss_definitions(objective):
    """Each algorithm owns its names; ordinary token-only protocols need no boilerplate."""
    definitions = getattr(objective, "loss_terms", None)
    return (
        definitions
        if definitions is not None
        else {"token": LossDefinition(objective.default_loss)}
    )


def _fields(payload, allowed, path):
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a mapping")
    unknown = payload.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown {path} fields: {sorted(unknown)}")


def resolve_loss_configuration(objective, payload):
    """Resolve mathematical terms and user-controlled execution settings.

    Returns:
        (LossConfiguration, LossExecution). The shared execution policy selects
        compatible optimizations only when loss.fused is true.
    """
    _fields(payload, {"terms", "fused", "scheduling"}, "loss")
    definitions = loss_definitions(objective)
    requested = payload.get("terms", {})
    _fields(requested, set(definitions), "loss.terms")
    resolved = {}
    for name, definition in definitions.items():
        options = requested.get(name, {})
        _fields(
            options,
            {"kind", "options", "coefficient"},
            f"loss.terms.{name}",
        )
        kind = options.get("kind", definition.loss.kind)
        formula_options = options.get(
            "options", definition.loss.options if kind == definition.loss.kind else {}
        )
        if not isinstance(formula_options, dict):
            raise ValueError(f"Options for loss term {name!r} must be a mapping")
        loss = TokenLoss(kind, dict(formula_options))
        if kind == "default":
            raise ValueError("Named loss terms require a concrete kind")
        if loss.registration.input_kind != definition.loss.registration.input_kind:
            raise ValueError(f"Loss {kind!r} is incompatible with {name!r} inputs")
        if definition.capabilities - loss.registration.capabilities:
            raise ValueError(f"Loss {kind!r} lacks required capabilities for {name!r}")
        coefficient = options.get("coefficient", definition.coefficient)
        if (
            type(coefficient) not in (int, float)
            or not math.isfinite(coefficient)
            or coefficient < 0
        ):
            raise ValueError(f"Invalid coefficient for loss term {name!r}")
        resolved[name] = ConfiguredLoss(loss, coefficient)
    scheduling = payload.get("scheduling", {})
    _fields(
        scheduling,
        {"chunk_size", "checkpoint", "trim", "teacher_chunk_size"},
        "loss.scheduling",
    )
    execution = LossExecution(fused=payload.get("fused", False), **scheduling)
    return LossConfiguration(resolved, execution.fused), execution
