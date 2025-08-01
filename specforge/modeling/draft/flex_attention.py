from torch.nn.attention.flex_attention import (
    or_masks,
    and_masks,
    create_block_mask,
    flex_attention,
)
import torch





def generate_eagle3_mask(seq_lengths: torch.Tensor, Q_LEN: int, KV_LEN: int):
    
    # TODO: Support batch size > 1, we need different suffix length for different batch

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def suffix_full(b, h, q_idx, kv_idx):
        return kv_idx >= Q_LEN

    def suffix_diagonal(b, h, q_idx, kv_idx):
        # Handles multiple occurance of like kv_idx - 2 * QLEN
        return (kv_idx - q_idx) % Q_LEN == 0
    
    def padding_mask_mod(b, h, q_idx, kv_idx):
        # Since right side padding
        return kv_idx < seq_lengths[b]

    first_half = and_masks(causal_mask, padding_mask_mod)
    second_half = and_masks(suffix_full, suffix_diagonal)
    mask_mod = or_masks(first_half, second_half)
    mask_mod.__name__ = f"eagle3_mask_Q_{Q_LEN}_KV_{KV_LEN}"
    return mask_mod

if __name__ == "__main__":
    # flex_attention = torch.compile(_flex_attention, dynamic=False)
    # flex_attention_call = lambda: flex_attention(query, key, value, block_mask=block_mask)
    B = 1
    H = 1
    S = 128 * 8
    Q_LEN = S
    KV_LEN = S * 4
    D = 128
    data_type = torch.bfloat16
    query = torch.randn(B, H, S, D, device="cuda", dtype=data_type, requires_grad=True)
    key_cache = torch.randn(B, H, KV_LEN, D, device="cuda", dtype=data_type, requires_grad=True)
    value_cache = torch.randn(B, H, KV_LEN, D, device="cuda", dtype=data_type, requires_grad=True)
    # Assuming first 128 tokens are padded
    padding_mask = torch.ones((B, S), device="cuda", dtype=torch.bool)
    padding_mask[:, 128:] = False
    padding_mask = padding_mask.unsqueeze(1).expand(B, S, S)
    seq_lengths = torch.tensor([128 * 4], device="cuda", dtype=torch.int32)

    block_mask = create_block_mask(
        mask_mod=generate_eagle3_mask(seq_lengths=seq_lengths, Q_LEN=Q_LEN, KV_LEN=KV_LEN), 
        B=1, 
        H=1,
        Q_LEN=Q_LEN, 
        KV_LEN=KV_LEN,
        device=query.device
    )

    dense_mask = block_mask.to_dense()
    print(block_mask.to_string())
    print(dense_mask)
    output = flex_attention(query, key_cache, value_cache, block_mask=block_mask)


