"""Resolve named loss settings for training wrappers awaiting full migration.

Wrappers keep their execution capabilities, but accept only the shared public
loss configuration. Unsupported formulas or scheduling fail before training.
"""

from dataclasses import asdict


def resolve_wrapper_loss(training, architecture, *, execution=()):
    """Return resolved terms/policy, rejecting controls this wrapper cannot honor."""
    from specforge.algorithms.objectives.registry import resolve_objective

    settings = resolve_objective(
        architecture,
        loss={"fused": True, **training.loss},
        weighting=training.weighting,
    )
    if settings.name in {"eagle3", "peagle"} and not settings.execution.fused:
        raise ValueError(
            f"{architecture} training wrapper requires training.loss.fused=true"
        )
    policy = asdict(settings.execution)
    for name, value in policy.items():
        if (
            name in training.loss.get("execution", {})
            and value
            and name not in execution
        ):
            raise ValueError(
                f"{architecture} training wrapper does not support loss.execution.{name}"
            )
    if settings.name in {"eagle3", "peagle"} and training.weighting is not None:
        if settings.weighting != resolve_objective(architecture).weighting:
            raise ValueError("This training wrapper uses fixed rollout weighting")
    _validate_terms(settings)
    return settings


def require_term(settings, name, kinds, *, options=(), coefficient=False):
    """Return a formula after checking the retained wrapper's supported inputs."""
    term = settings.losses.terms[name]
    if (
        term.loss.kind not in kinds
        or term.loss.options.keys() - set(options)
        or (not coefficient and term.coefficient != 1)
    ):
        raise ValueError(
            f"{settings.name} training wrapper cannot execute loss.terms.{name}={term}; "
            "use the canonical training path for this loss"
        )
    return term


def token_arguments(settings, *, teacher=False):
    """Return the scalar arguments consumed by the retained CE/LK kernels."""
    default = "soft_ce" if teacher else "ce"
    term = require_term(
        settings,
        "token",
        {default, "lk_alpha", "lk_tv", "lk_lambda"},
        options={"target", "kl_scale", "kl_decay"},
    )
    loss = term.loss
    if loss.kind != default and loss.options.get("target", "labels") != (
        "teacher" if teacher else "labels"
    ):
        raise ValueError("Loss target does not match the training wrapper's inputs")
    return {
        "lk_loss_type": loss.kind.removeprefix("lk_") if loss.kind != default else None,
        "kl_scale": loss.options.get("kl_scale", 1.0),
        "kl_decay": loss.options.get("kl_decay", 1.0),
    }


def _validate_terms(settings):
    """Check formulas and weighting supported by retained wrapper kernels."""
    name = settings.name
    if name == "eagle3":
        token_arguments(settings, teacher=True)
    elif name == "peagle":
        require_term(settings, "token", {"soft_ce"})
    elif name in {"dflash", "dflash2"}:
        token_arguments(settings)
        if name == "dflash2":
            require_term(settings, "selector", {"ce"}, coefficient=True)
    elif name == "domino":
        for term in ("base", "final"):
            require_term(settings, term, {"ce"})
    elif name == "dspark":
        require_term(settings, "token", {"ce_l1"}, options={"ce_weight", "l1_weight"})
        require_term(settings, "confidence", {"bce"}, coefficient=True)
    elif name == "dspine":
        token = require_term(
            settings, "token", {"ce_l1"}, options={"ce_weight", "l1_weight"}
        )
        refinement = require_term(
            settings, "refinement", {"ce_l1"}, options={"ce_weight", "l1_weight"}
        )
        require_term(settings, "alignment", {"cosine"}, coefficient=True)
        if (
            token.loss != refinement.loss
            or abs(sum(token.loss.options.values()) - 1) > 1e-8
        ):
            raise ValueError(
                "DSpine wrapper requires matching token/refinement weights summing to one"
            )
    if name not in {"eagle3", "peagle"}:
        supported = {"uniform", "exponential"}
        if name in {"dflash", "dflash2"}:
            supported |= {
                "dpace",
                "dpace-cumulative-confidence-only",
                "dpace-continuation-value-only",
            }
        if settings.weighting.kind not in supported:
            raise ValueError(
                "This training wrapper does not support the requested weighting"
            )


def validate_dspine_weights(training, recorded):
    """Require explicit replacement of nondefault weights in historical drafts."""
    terms = training.loss.get("terms", {})
    ce_weight = recorded.get("ce_weight", 0.1)
    if ce_weight != 0.1:
        for name in ("token", "refinement"):
            options = terms.get(name, {}).get("options", {})
            if not {"ce_weight", "l1_weight"} <= options.keys():
                raise ValueError(
                    f"Draft records ce_weight={ce_weight}; explicitly set "
                    f"training.loss.terms.{name}.options.ce_weight and l1_weight"
                )
    alignment = recorded.get("alignment_weight", 0.5)
    if alignment != 0.5 and "coefficient" not in terms.get("alignment", {}):
        raise ValueError(
            f"Draft records alignment_weight={alignment}; explicitly set "
            "training.loss.terms.alignment.coefficient"
        )


def wrapper_loss_validator(architecture, *, execution=(), compatible=()):
    """Build a planning hook using the same loss resolver as model assembly.

    Reuse model loading's config source resolution to distinguish DFlash2 and
    historical weights, without loading target weights or constructing a model.
    """

    def validate(config):
        from specforge.training.model_loading import load_explicit_draft_config

        draft = load_explicit_draft_config(config)
        recorded = draft.to_dict() if draft is not None else {}
        selected = recorded.get("architectures", [architecture])[0]
        if selected not in (architecture, *compatible):
            selected = architecture
        resolve_wrapper_loss(config.training, selected, execution=execution)
        if selected == "DSpineDraftModel":
            validate_dspine_weights(config.training, recorded.get("dspine_config", {}))

    return validate
