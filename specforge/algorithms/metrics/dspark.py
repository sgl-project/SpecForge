"""Detached DSpark acceptance and teacher-distribution diagnostics."""

import torch.nn.functional as F


def measure(outputs, labels, mask, teacher, weights):
    """Return additive per-position/teacher statistics and hard-label tokens."""
    logits = outputs["logits"].float()
    probabilities = logits.softmax(-1)
    ids = logits.argmax(-1)
    valid = mask > 0
    denominator = valid.sum().float()
    ce = F.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), reduction="none"
    ).view_as(mask)
    ratios = {
        "ce_loss": ((ce * weights).sum(), weights.sum()),
        "ce_position": ((ce * valid).sum(0), valid.sum(0).float()),
        "accuracy_position": (
            ((ids == labels) & valid).sum(0).float(),
            valid.sum(0).float(),
        ),
    }
    if teacher is not None:
        target = teacher.float().softmax(-1)
        acceptance = (1 - (probabilities - target).abs().sum(-1) / 2).clamp(0, 1)
        distance = (probabilities - target).abs().sum(-1)
        ratios["l1_loss"] = ((distance * weights).sum(), weights.sum())
        confidence = outputs.get("confidence_scores")
        if confidence is not None:
            error = (confidence.float().sigmoid() - acceptance).abs()
            loss = F.binary_cross_entropy_with_logits(
                confidence.float(), acceptance, reduction="none"
            )
            ratios["confidence_abs_error"] = ((error * weights).sum(), weights.sum())
            ratios["confidence_loss"] = ((loss * weights).sum(), weights.sum())
        blocks = valid.any(-1)
        ratios.update(
            {
                "teacher_agreement": (
                    ((ids == target.argmax(-1)) & valid).sum().float(),
                    denominator,
                ),
                "teacher_top1_prob": ((target.amax(-1) * valid).sum(), denominator),
                "draft_top1_prob": (
                    (probabilities.amax(-1) * valid).sum(),
                    denominator,
                ),
                "tau_probabilistic": (
                    ((acceptance * valid).cumprod(-1).sum(-1) + 1).mul(blocks).sum(),
                    blocks.sum().float(),
                ),
            }
        )
    return ratios, {"dspark/hard_label": ids}
