"""Named loss terms and joint execution, resolved before training starts."""

import math
from dataclasses import asdict, dataclass, field, replace

from .execution_options import LossExecution
from .losses import TokenLoss


@dataclass(frozen=True)
class LossMode:
    """Implementation choice for one term or group, independent of scheduling."""

    fused: bool = False
    impl: str | None = None

    def __post_init__(self):
        if type(self.fused) is not bool:
            raise ValueError("fused must be boolean")
        if self.impl is not None:
            if not isinstance(self.impl, str) or not self.impl:
                raise ValueError("impl must be a nonempty name")
            if self.fused:
                raise ValueError("impl and fused=True are mutually exclusive")


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
    execution: LossMode = field(default_factory=LossMode)


@dataclass(frozen=True)
class LossGroup:
    terms: tuple[str, ...]
    execution: LossMode = field(default_factory=LossMode)


@dataclass(frozen=True)
class LossConfiguration:
    """Resolved names, formulas and execution choices; no model tensors."""

    terms: dict[str, ConfiguredLoss]
    groups: dict[str, LossGroup]

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
                execution=self.terms[name].execution,
            )
            for name, term in zip(names, plan.terms)
        )
        groups = tuple(
            tuple(names.index(name) for name in group.terms)
            for group in self.groups.values()
        )
        modes = {
            indices: group.execution
            for indices, group in zip(groups, self.groups.values())
        }
        return replace(plan, terms=terms, joint_groups=groups, group_execution=modes)

    def to_dict(self):
        """Stable effective settings persisted in the resume contract."""
        grouped = {name for group in self.groups.values() for name in group.terms}
        return {
            "terms": {
                name: {
                    **asdict(term.loss),
                    "coefficient": term.coefficient,
                    **(
                        {"execution": asdict(term.execution)}
                        if name not in grouped
                        else {}
                    ),
                }
                for name, term in self.terms.items()
            },
            "groups": {
                name: {"terms": list(group.terms), "execution": asdict(group.execution)}
                for name, group in self.groups.items()
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


def _mode(payload, path):
    _fields(payload, {"fused", "impl"}, path)
    return LossMode(**payload)


def resolve_loss_configuration(objective, payload):
    """Resolve defaults and reject unknown terms, incompatible inputs and overlaps.

    Returns:
        (LossConfiguration, LossExecution). Scheduling contains no implementation
        choice; an omitted per-term/group execution always means reference.
    """
    _fields(payload, {"terms", "groups", "scheduling"}, "loss")
    definitions = loss_definitions(objective)
    requested = payload.get("terms", {})
    _fields(requested, set(definitions), "loss.terms")
    resolved = {}
    for name, definition in definitions.items():
        options = requested.get(name, {})
        _fields(
            options,
            {"kind", "options", "coefficient", "execution"},
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
        mode = _mode(options.get("execution", {}), f"loss.terms.{name}.execution")
        _validate_mode(mode, (loss,))
        resolved[name] = ConfiguredLoss(loss, coefficient, mode)
    groups = {}
    occupied = set()
    raw_groups = payload.get("groups", {})
    if not isinstance(raw_groups, dict):
        raise ValueError("loss.groups must be a mapping")
    for name, group in raw_groups.items():
        _fields(group, {"terms", "execution"}, f"loss.groups.{name}")
        members = group.get("terms", [])
        if (
            not isinstance(members, (list, tuple))
            or len(members) < 2
            or any(not isinstance(v, str) for v in members)
        ):
            raise ValueError("A loss group requires at least two named terms")
        if (
            len(set(members)) != len(members)
            or set(members) - resolved.keys()
            or occupied.intersection(members)
        ):
            raise ValueError(
                f"Unknown, repeated or overlapping terms in group {name!r}"
            )
        allowed = getattr(objective, "joint_loss_groups", ())
        if not any(set(members) == set(group) for group in allowed):
            raise ValueError(
                f"Algorithm does not provide joint inputs for group {name!r}"
            )
        if any("execution" in requested.get(member, {}) for member in members):
            raise ValueError("Group members cannot specify individual execution")
        mode = _mode(group.get("execution", {}), f"loss.groups.{name}.execution")
        _validate_mode(mode, tuple(resolved[member].loss for member in members))
        groups[name] = LossGroup(tuple(members), mode)
        occupied.update(members)
    scheduling = payload.get("scheduling", {})
    _fields(
        scheduling,
        {"chunk_size", "checkpoint", "trim", "teacher_chunk_size"},
        "loss.scheduling",
    )
    return LossConfiguration(resolved, groups), LossExecution(**scheduling)


def _validate_mode(mode, losses):
    if mode.impl is None:
        return
    from .implementations.registry import implementation_registrations

    entry = implementation_registrations().get(mode.impl)
    if entry is None or entry.category != "specialized":
        raise ValueError(
            f"{mode.impl!r} is not a registered specialized implementation"
        )
    if entry.losses and any(loss.kind not in entry.losses for loss in losses):
        raise ValueError(
            f"{mode.impl} does not implement the selected mathematical losses"
        )
