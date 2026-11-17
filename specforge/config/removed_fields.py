"""Removed configuration fields and the settings that replace them.

Nothing is converted here: a removed field is rejected with its replacement.
"""

REMOVED_LOSS_FIELDS = {
    "fused_logits_loss": "loss.fused",
    "lk_loss_type": "loss.terms.token.kind (lk_alpha, lk_tv, lk_lambda); omit for CE",
    "kl_scale": "loss.terms.token.options.kl_scale with kind=lk_lambda",
    "kl_decay": "loss.terms.token.options.kl_decay with kind=lk_lambda",
    "loss_decay_gamma": "weighting.kind=exponential and weighting.gamma; use uniform for nonpositive gamma",
    "loss_type": "weighting.kind (uniform, exponential, or a dpace variant)",
    "dpace_alpha": "weighting.alpha with a dpace variant",
    "objective_chunk_blocks": "loss.execution.chunk_size",
    "mtp_objective_chunk_size": "loss.execution.chunk_size",
    "objective_checkpointing": "loss.execution.checkpoint",
    "trim_loss_positions": "loss.execution.trim",
    "compact_teacher": "loss.execution.teacher_chunk_size (positive enables; omit disables)",
    "compact_teacher_chunk_size": "loss.execution.teacher_chunk_size",
    "dflash2_selector_loss_alpha": "loss.terms.selector.coefficient",
    "dspark_ce_loss_alpha": "loss.terms.token.options.ce_weight",
    "dspark_l1_loss_alpha": "loss.terms.token.options.l1_weight",
    "dspark_confidence_head_alpha": "loss.terms.confidence.coefficient",
}


def reject_removed_loss_fields(fields, *, override=False):
    """Reject retired keys with their replacement paths.

    Args:
        fields: Names present under ``training``.
        override: The names come from a command-line override, which cannot
            create a nested mapping the file lacks; the message then shows the
            mapping form.
    """
    removed = [
        f"training.{key} -> training.{replacement}"
        for key, replacement in REMOVED_LOSS_FIELDS.items()
        if key in fields
    ]
    if removed:
        message = "Removed training loss fields; replace explicitly: " + "; ".join(
            removed
        )
        if override:
            message += (
                ". As an override, pass the enclosing mapping unless the file "
                "already has it, for example "
                "'training.loss={execution: {chunk_size: 64}}'"
            )
        raise ValueError(message)
