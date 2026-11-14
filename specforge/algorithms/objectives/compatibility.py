"""Translate previous loss settings once, before constructing an objective."""

from copy import deepcopy
from dataclasses import asdict

from .execution_options import LossExecution
from .losses import TokenLoss

# Historical names are confined to this input adapter. Runtime protocols only
# receive mathematical terms and the shared execution policy.
_OPTIONS = {
    "dspark": {
        "ce_loss_alpha": (("token", "options", "ce_weight"),),
        "l1_loss_alpha": (("token", "options", "l1_weight"),),
        "confidence_loss_alpha": (("confidence", "coefficient"),),
    },
    "dspine": {
        "ce_weight": (
            ("token", "options", "ce_weight"),
            ("refinement", "options", "ce_weight"),
        ),
        "alignment_weight": (("alignment", "coefficient"),),
    },
    "dflash2": {
        "selector_loss_alpha": (("selector", "coefficient"),),
    },
    "domino": {"impl": ()},
}


def normalize_loss_settings(name, options, loss, execution):
    """Return algorithm options and one named loss payload.

    Old flat formulas and scheduling are converted without a second execution
    path. Mixing old loss options with named settings is rejected.
    """
    options, loss = dict(options or {}), deepcopy(loss or {})
    mappings = _OPTIONS.get(name, {})
    old_options = options.keys() & mappings.keys()
    named = bool({"terms", "fused", "groups", "scheduling"} & loss.keys())
    if named:
        if execution:
            raise ValueError(
                "Use loss.fused and loss.scheduling instead of loss_execution"
            )
        if old_options:
            raise ValueError(
                "Configure loss definitions and coefficients under loss.terms"
            )
        return options, loss

    formula = TokenLoss(**loss)
    policy = dict(execution or {})
    LossExecution(**policy)  # Validate old options before translating them.
    payload = {"terms": {}, "fused": policy.pop("fused", False), "scheduling": policy}
    primary = ("base", "final") if name == "domino" else ("token",)
    if formula.kind != "default":
        if old_options & {"ce_loss_alpha", "l1_loss_alpha"}:
            raise ValueError(
                f"{sorted(old_options)} only configure the default loss; use loss.terms"
            )
        payload["terms"] = {term: deepcopy(loss) for term in primary}
    for key in old_options:
        value = options.pop(key)
        if key == "impl":
            if value is not None:
                if value != "domino_suffix_ce":
                    raise ValueError(
                        "Explicit loss implementations are no longer supported"
                    )
                payload["fused"] = True
            continue
        for path in mappings[key]:
            if formula.kind != "default" and name == "dspine" and path[0] == "token":
                continue
            target = payload["terms"]
            for part in path[:-1]:
                target = target.setdefault(part, {})
            target[path[-1]] = value
            if name == "dspine" and key == "ce_weight":
                target["l1_weight"] = 1 - value
    return options, payload


def normalize_loss_checkpoint(state, objective_name):
    """Normalize historical canonical loss metadata without changing model state.

    Only known equivalent execution identities are translated. Unknown changes
    remain subject to the ordinary resume checks.
    """
    from .configuration import resolve_loss_configuration
    from .registry import protocol_registrations

    if "canonical_objective_options" not in state:
        return state
    state = dict(state)
    name = state.get("canonical_objective", objective_name)
    loss = deepcopy(state.get("canonical_loss", {}))
    execution = state.get("canonical_loss_execution", {})
    named = bool({"terms", "fused", "scheduling"} & loss.keys())
    options, inherited = normalize_loss_settings(
        name, state["canonical_objective_options"], None, execution
    )
    if named:
        # Named formulas already contain effective coefficients; the old
        # objective fields are redundant defaults in these checkpoints.
        if (
            "fused" in loss
            and "fused" in execution
            and loss["fused"] != execution["fused"]
        ):
            raise ValueError("Checkpoint records conflicting loss fusion settings")
        payload = loss | {
            "fused": loss.get("fused", inherited["fused"]),
            "scheduling": inherited["scheduling"],
        }
    else:
        payload = inherited
        formula = TokenLoss(**loss)
        if formula.kind != "default":
            primary = ("base", "final") if name == "domino" else ("token",)
            for term in primary:
                payload["terms"][term] = asdict(formula)
    objective = protocol_registrations()[name].factory(**options)
    losses, policy = resolve_loss_configuration(objective, payload)
    state["canonical_objective_options"] = asdict(objective)
    if "canonical_loss" in state:
        state["canonical_loss"] = losses.to_dict()
    if "canonical_loss_execution" in state:
        state["canonical_loss_execution"] = asdict(policy)
    if name == "dspine" and not named and "canonical_loss_implementations" in state:
        identities = dict(state["canonical_loss_implementations"])
        for key, implementation in tuple(identities.items()):
            if key.endswith("/compute/refinement:0"):
                if implementation == "token_ce":
                    identities[key] = "token_ce_l1"
                # The same cosine formula previously bypassed fused selection.
                prefix = key.split("/compute/")[0]
                identities[prefix + "/compute/alignment:0"] = "reference"
        state["canonical_loss_implementations"] = identities
    return state
