"""Detached diagnostics of rollout drafts against the supervision of their loss."""

import torch


def expected_acceptance(predictions, outputs, supervision):
    """Return the additive expected acceptance of one chunk and its positions.

    A position accepts with probability sum(min(p, q)) over the draft
    vocabulary: q is the draft distribution and p the teacher's probability of
    each draft token under its full vocabulary.
    """
    draft = outputs["logits"].float().softmax(-1)
    teacher = supervision.teacher.float().softmax(-1)
    if supervision.teacher_mass is not None:
        teacher = teacher * supervision.teacher_mass[..., None]
    mask = supervision.mask.float()
    accepted = torch.minimum(teacher, draft).sum(-1)
    return {"accepted": (accepted * mask).sum(), "supervised": mask.sum()}


def depth_accuracy(predictions, outputs, supervision, *, depths):
    """Return one chunk's correct and supervised position counts by depth."""
    supervised = supervision.mask > 0
    correct = (outputs["logits"].argmax(-1) == supervision.labels) & supervised
    depth = predictions.positions.flatten()
    counts = torch.zeros(depths, device=depth.device)
    return {
        "correct": counts.index_add(0, depth, correct.flatten().float()),
        "supervised": counts.index_add(0, depth, supervised.flatten().float()),
    }
