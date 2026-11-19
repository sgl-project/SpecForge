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


#: Removed with the legacy training entry. Each maps to where the setting lives
#: now; "draft" means the draft configuration (a preset path or inline mapping).
REMOVED_FIELDS = {
    "model": {
        "draft_model_config": "draft (a preset under specforge/config/draft/presets, or an inline canonical configuration)",
        "draft_checkpoint_path": "initialization.checkpoint to import weights, or resume for a training checkpoint",
        "draft_num_hidden_layers": "the layer list of the draft configuration",
        "draft_block_size": "spec_config.block.block_size of the draft configuration",
        "mask_token_id": "spec_config of the draft configuration (the presets carry it)",
        "aux_hidden_state_layer_ids": "target_features.target_layer_ids of the draft configuration",
    },
    "training": {
        "strategy": "draft (its architecture selects the objective; training.objective chooses another)",
        "num_anchors": "training.objective_options.num_anchors",
        "ttt_length": "training.objective_options.ttt_length",
        "num_depths": "training.objective_options.num_depths",
        "down_sample_ratio": "training.objective_options.down_sample_ratio",
        "down_sample_ratio_min": "training.objective_options.down_sample_ratio_min",
        "lambda_base_start": "training.objective_options.lambda_base_start",
        "lambda_base_decay_ratio": "training.objective_options.lambda_base_decay_ratio",
        "dflash2_selector_warmup_ratio": "training.objective_options.selector_warmup_ratio",
        "dflash2_selector_ramp_ratio": "training.objective_options.selector_ramp_ratio",
        "dflash2_selector_stop_gradient": "training.objective_options.selector_stop_gradient",
        "dflash_teacher_metrics": "training.objective_options.teacher_metrics",
        "norm_before_residual": "spec_config.norm_before_residual of the draft configuration",
        "deployment_mode": "deployment.mode",
        "server_urls": "deployment.disaggregated.server_urls",
        "metadata_db_path": "deployment.disaggregated.consumer_state_dir (the directory that held consumer.sqlite)",
    },
    "initialization": {
        "legacy_config": "scripts/convert_legacy_checkpoint.py --draft-config, which writes a canonical checkpoint; only those can be imported",
    },
}


def reject_removed_fields(section, fields):
    """Reject keys removed from one config section, naming each replacement.

    Args:
        section: "model", "training" or "initialization".
        fields: Names present in that section.
    """
    removed = [
        f"{section}.{key} -> {replacement}"
        for key, replacement in REMOVED_FIELDS.get(section, {}).items()
        if key in fields
    ]
    if removed:
        raise ValueError("Removed configuration fields: " + "; ".join(removed))


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
