import torch
from torch.nn.attention.flex_attention import create_block_mask

def create_ttt_mask(mask, bsz: int, seq_len: int, ttt_step: int):
    """
    A corrected implementation to create the special TTT attention mask using the flex_attention API.
    This version uses positional arguments for create_block_mask for maximum compatibility.

    Args:
        mask: not used yet
        bsz (int): Batch size.
        seq_len (int): The sequence length of each block (k).
        ttt_step (int): Represents the index of the current additional block (starting from 0). 
                        The total number of blocks is ttt_step + 1.
    """
    # Calculate the total length of the KV cache
    total_kv_len = seq_len * (ttt_step + 1)

    def combined_mask_fn(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ):
        """
        This function is the core of flex_attention, defining all attention rules.
        """
        # Condition 1: If the key is in the first block, apply a causal mask.
        first_block_mask = (q_idx >= kv_idx)
        
        # Condition 2: If the key is in subsequent blocks, apply a diagonal mask.
        subsequent_blocks_mask = (q_idx == (kv_idx % seq_len))
        
        # Combine the two conditions with a logical OR.
        return first_block_mask | subsequent_blocks_mask
    
    # Call create_block_mask using only positional arguments in the correct order
    # for older PyTorch versions: (mask_fn, bsz, q_len, kv_len)
    block_mask = create_block_mask(
        combined_mask_fn,
        bsz,
        None,
        seq_len,
        total_kv_len
    )
    
    return block_mask

# --- Test Script ---
if __name__ == "__main__":
    bsz_test = 1
    seq_len_test = 512  # Equivalent to 'k'
    ttt_step_test = 2 # ttt_step=2 means there are 2+1=3 blocks in total.
    
    print(f"--- Testing Flex Attention Mask (Corrected) ---")
    print(f"Parameters: bsz={bsz_test}, seq_len(k)={seq_len_test}, ttt_step={ttt_step_test} (total blocks i=3)\n")

    # Create the mask object
    ttt_mask_object = create_ttt_mask(
        bsz=bsz_test,
        seq_len=seq_len_test,
        ttt_step=ttt_step_test
    )

    print(f"Successfully created flex_attention.BlockMask object: {ttt_mask_object}")
    print("This is an efficient object representation, not a dense tensor.")
    print("To verify its correctness, we can 'materialize' it into a dense tensor for inspection.\n")

    print(ttt_mask_object)