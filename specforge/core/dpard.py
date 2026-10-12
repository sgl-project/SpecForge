"""D-PARD: detached TV position credit with a Rényi-half local objective."""

import torch
import torch.nn.functional as functional


def dpard_token_terms(draft_logits, target_logits, valid_mask, rho=0.5):
    """Return per-position Rényi-half divergence and detached survival credit.

    Positions occupy the penultimate logits axis. Invalid positions neither
    change cumulative survival nor contribute to the continuation sum.
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError("D-PARD rho must be in [0, 1]")
    if draft_logits.shape != target_logits.shape:
        raise ValueError("D-PARD draft and target logits must have identical shapes")
    if draft_logits.shape[:-1] != valid_mask.shape:
        raise ValueError("D-PARD valid_mask must match the token dimensions")
    log_draft = functional.log_softmax(draft_logits.float(), dim=-1)
    log_target = functional.log_softmax(target_logits.detach().float(), dim=-1)
    renyi = -2.0 * torch.logsumexp(0.5 * (log_target + log_draft), dim=-1)
    with torch.no_grad():
        acceptance = torch.minimum(log_target.exp(), log_draft.exp()).sum(dim=-1)
        survival = rho + (1.0 - rho) * acceptance
        survival = torch.where(valid_mask, survival, torch.ones_like(survival))
        reach = survival.cumprod(dim=-1) * valid_mask
        credit = reach.flip(-1).cumsum(-1).flip(-1) * valid_mask
    return renyi, credit
