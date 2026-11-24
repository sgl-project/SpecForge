"""Log-space objectives for aligned teacher and draft vocabulary logits."""

import torch


def renyi_half(logits, teacher):
    log_p = teacher.float().log_softmax(-1)
    log_q = logits.float().log_softmax(-1)
    return -2 * torch.logsumexp((log_p + log_q) * 0.5, dim=-1)


def neg_log_overlap(logits, teacher):
    # Stay in log space even when the two distributions barely overlap.
    log_p = teacher.float().log_softmax(-1)
    log_q = logits.float().log_softmax(-1)
    return -torch.logsumexp(torch.minimum(log_p, log_q), dim=-1)
