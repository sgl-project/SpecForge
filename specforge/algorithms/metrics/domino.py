"""Detached diagnostics for base and corrected Domino predictions."""

import torch.nn.functional as F


def measure(outputs, labels, mask, teacher, weights, *, base_weight):
    """Return additive accuracies/accepted lengths and tokens for anchor walks."""
    denominator = weights.sum()
    ratios = {"lambda_base": (base_weight * denominator, denominator)}
    tokens = {}
    valid = mask > 0
    for name, key in (("final", "logits"), ("base", "base_logits")):
        logits = outputs[key].float()
        ids = logits.argmax(-1)
        ce = F.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), reduction="none"
        ).view_as(mask)
        ratios[f"{name}_loss"] = ((ce * weights).sum(), weights.sum())
        correct = ids == labels
        prefix = (correct | ~valid).long().cumprod(-1) * valid
        blocks = valid.any(-1)
        ratios["acc" if name == "final" else "base_accuracy"] = (
            (correct & (mask > 0.5)).sum().float(),
            mask.sum(),
        )
        ratios["accept_len" if name == "final" else "base_accept_len"] = (
            ((prefix.sum(-1) + 1) * blocks).sum().float(),
            blocks.sum().float(),
        )
        tokens[f"domino/{name}"] = ids
    return ratios, tokens
