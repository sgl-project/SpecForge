"""D-PAL local losses and detached target-draft overlap (arXiv:2610.28355)."""

import torch
import torch.nn.functional as F

DPAL_LOSS_TYPES = frozenset({"dpal-renyi", "dpal-kl", "dpal-la", "dpal-ce-tv"})


@torch.compile(dynamic=True, disable=not torch.cuda.is_available())
def dpal_actor_and_acceptance(logits, target_logits, loss_type, target_ids=None):
    log_q = F.log_softmax(logits.float(), dim=-1)
    log_p = F.log_softmax(target_logits.detach().float(), dim=-1)
    log_overlap = torch.logsumexp(torch.minimum(log_p, log_q), dim=-1)
    if loss_type == "dpal-renyi":
        actor = -2.0 * torch.logsumexp(0.5 * (log_p + log_q), dim=-1)
    elif loss_type == "dpal-kl":
        p = log_p.exp()
        actor = torch.where(p > 0, p * (log_p - log_q), 0.0).sum(-1)
    elif loss_type == "dpal-la":
        actor = -log_overlap
    elif loss_type == "dpal-ce-tv":
        if target_ids is None:
            raise ValueError("dpal-ce-tv requires recorded target token IDs")
        ce = -log_q.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        tv = 0.5 * (log_p.exp() - log_q.exp()).abs().sum(-1)
        actor = 0.1 * ce + 0.9 * tv
    else:
        raise ValueError(f"unsupported D-PAL loss: {loss_type!r}")
    return actor, log_overlap.detach().exp().clamp(0.0, 1.0)
