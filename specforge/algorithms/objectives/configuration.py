"""Named mathematical losses and a shared optimization policy."""

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
    """Resolved named formulas and coefficients; no execution policy or tensors."""

    terms: dict[str, ConfiguredLoss]

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


def resolve_loss_configuration(objective, payload):
    """Validate loss syntax, then bind formulas to the algorithm's input contracts."""
    from pydantic import ValidationError

    from specforge.config.schema import LossConfig

    try:
        parsed = LossConfig.model_validate(payload)
    except ValidationError as error:
        details = "; ".join(
            f"training.loss.{'.'.join(map(str, item['loc']))}={item['input']!r}: {item['msg']}"
            for item in error.errors()
        )
        raise ValueError(details) from error
    definitions = loss_definitions(objective)
    unknown = parsed.terms.keys() - definitions.keys()
    if unknown:
        raise ValueError(f"Unknown training.loss.terms: {sorted(unknown)!r}")
    resolved = {}
    for name, definition in definitions.items():
        options = (
            parsed.terms[name].model_dump(exclude_unset=True)
            if name in parsed.terms
            else {}
        )
        path = f"training.loss.terms.{name}"
        kind = options.get("kind", definition.loss.kind)
        defaults = definition.loss.options if kind == definition.loss.kind else {}
        formula_options = {**defaults, **options.get("options", {})}
        try:
            loss = TokenLoss(kind, formula_options)
        except ValueError as error:
            # Include exact option paths without duplicating formula validators here.
            values = ", ".join(
                f"{path}.options.{key}={value!r}"
                for key, value in formula_options.items()
            )
            raise ValueError(f"{path}.kind={kind!r}; {values}: {error}") from error
        if kind == "default":
            raise ValueError(f"{path}.kind={kind!r}: a concrete formula is required")
        if loss.registration.input_kind != definition.loss.registration.input_kind:
            raise ValueError(f"{path}.kind={kind!r}: incompatible with {name!r} inputs")
        if definition.capabilities - loss.registration.capabilities:
            raise ValueError(f"{path}.kind={kind!r}: lacks required capabilities")
        resolved[name] = ConfiguredLoss(
            loss, options.get("coefficient", definition.coefficient)
        )
    execution = LossExecution(fused=parsed.fused, **parsed.execution.model_dump())
    return LossConfiguration(resolved), execution
