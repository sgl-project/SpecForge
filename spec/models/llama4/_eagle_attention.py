import torch

def create_ttt_mask(seq_len, cache_lengths):
    """
    Create TTT special attention mask
    cache_lengths: [original_len, step1_len, step2_len, ...]
    """
    def score_mod(score, b, h, q_idx, kv_idx):
        cumsum_lengths = torch.cumsum(torch.tensor(cache_lengths), dim=0)
        
        # Determine which segment current q_idx and kv_idx belong to
        q_segment = torch.searchsorted(cumsum_lengths, q_idx, right=True)
        kv_segment = torch.searchsorted(cumsum_lengths, kv_idx, right=True)
        
        # TTT mask rules
        if q_segment == kv_segment == 0:
            # Original training data segment: standard causal mask
            return score if q_idx >= kv_idx else float('-inf')
        elif q_segment > 0 and kv_segment == 0:
            # Draft steps can attend to original data
            return score
        elif q_segment > 0 and kv_segment > 0:
            # Between draft steps: only attend to corresponding positions
            relative_q = q_idx - cumsum_lengths[q_segment-1]
            relative_kv = kv_idx - cumsum_lengths[kv_segment-1]
            return score if relative_q == relative_kv else float('-inf')
        else:
            return float('-inf')
    
    return score_mod