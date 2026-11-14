"""Resolve named loss settings for training wrappers awaiting full migration.

Wrappers keep their execution capabilities, but accept only the shared public
loss configuration. Unsupported formulas or scheduling fail before training.
"""

from dataclasses import asdict

from specforge.algorithms.objectives.registry import resolve_objective


def resolve_wrapper_loss(training, architecture, *, scheduling=()):
    """Return resolved terms/policy, rejecting controls this wrapper cannot honor."""
    settings = resolve_objective(
        architecture, loss=training.loss, weighting=training.weighting
    )
    policy = asdict(settings.execution)
    for name, value in policy.items():
        if (
            name in training.loss.get("scheduling", {})
            and value
            and name not in scheduling
        ):
            raise ValueError(
                f"{architecture} training wrapper does not support loss.scheduling.{name}"
            )
    if settings.name in {"eagle3", "peagle"} and training.weighting is not None:
        if settings.weighting != resolve_objective(architecture).weighting:
            raise ValueError("This training wrapper uses fixed rollout weighting")
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
