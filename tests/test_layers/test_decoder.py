import os
import unittest

import torch
import torch.multiprocessing as mp
from accelerate.utils import set_seed
from torch import nn
from transformers import PretrainedConfig
from yunchang import EXTRACT_FUNC_DICT

# Project-specific imports
from specforge.distributed import destroy_distributed, init_distributed
from specforge.modeling.draft.llama3_eagle import LlamaDecoderLayer
from specforge.utils import padding
from tests.utils import get_available_port


def get_model_config():
    """Create and return the model configuration."""
    config_dict = {
        "architectures": ["LlamaForCausalLMEagle3"],
        "eagle_config": {
            "eagle_aux_hidden_state_layer_ids": [1, 29, 57],
            "use_aux_hidden_state": True,
        },
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "hidden_act": "silu",
        "hidden_size": 7168,
        "initializer_range": 0.02,
        "intermediate_size": 29568,
        "max_position_embeddings": 32768,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 1,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-05,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "transformers_version": "4.28.1",
        "use_cache": True,
        "rope_scaling": None,
        "vocab_size": 129280,
        "draft_vocab_size": 32000,
        "pretraining_tp": 1,
    }
    return PretrainedConfig.from_dict(config_dict)


def setup_env(rank, world_size, port):
    """Set up distributed environment variables."""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)


def run_iterative_pass(
    decoder_layer,
    embed_tokens,
    input_ids,
    hidden_states,
    attention_mask,
    position_ids,
    ttt_length,
):
    """
    Core loop: execute the forward pass `ttt_length` times.
    Used for both Golden (SDPA) and Distributed (USP) runs to ensure logic consistency.
    """
    # Clone to avoid side effects on original tensors
    curr_input_ids = input_ids.clone()
    curr_hidden_states = hidden_states.clone()

    # Init cache
    cache_hidden = [[], []]
    past_key_values = None
    final_output = None

    for idx in range(ttt_length):
        is_last = idx == ttt_length - 1

        # 1. Embed inputs
        inputs_embeds = embed_tokens(curr_input_ids).to(curr_hidden_states.dtype)

        # 2. Forward pass
        output_hidden_states = decoder_layer(
            input_emb=inputs_embeds,
            hidden_states=curr_hidden_states,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=False,
            use_cache=False,
        )

        # Update states for next iteration
        curr_hidden_states = output_hidden_states
        final_output = output_hidden_states

        # 3. Simulate TTT padding/shift
        if not is_last:
            curr_input_ids = padding(curr_input_ids, left=False)

    return final_output


def run_test_case(rank, world_size, port):
    """Worker function executed in each process."""
    setup_env(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")
    set_seed(42)

    # --- Data & Config Preparation ---
    config = get_model_config()
    seq_len = 1560
    batch_size = 1
    ttt_length = 3

    # Generate dummy data on GPU
    data_input_ids = torch.randint(0, 10000, (batch_size, seq_len), device=device)
    data_hidden_states = torch.randn(
        batch_size, seq_len, config.hidden_size, device=device, dtype=torch.bfloat16
    )
    attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).view(
        1, 1, seq_len, seq_len
    )
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # Shared embedding layer
    embed_tokens = nn.Embedding(
        config.vocab_size, config.hidden_size, config.pad_token_id
    ).to(device)

    # --- Phase 1: Golden Run (SDPA) ---
    # Init dist briefly for internal checks, even if running single-device logic
    init_distributed(tp_size=1, sp_ulysses_size=1, sp_ring_size=1)

    sdpa_decoder = (
        LlamaDecoderLayer(config, attention_backend="fa").to(device).to(torch.bfloat16)
    )

    with torch.no_grad():
        sdpa_output = run_iterative_pass(
            decoder_layer=sdpa_decoder,
            embed_tokens=embed_tokens,
            input_ids=data_input_ids,
            hidden_states=data_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            ttt_length=ttt_length,
        )

    # Save weights for alignment and cleanup SDPA model
    state_dict = sdpa_decoder.state_dict()
    del sdpa_decoder
    destroy_distributed()

    # --- Phase 2: Distributed Run (USP) ---
    def subtest_usp(sp_ulysses_degree, sp_ring_degree):
        """Run USP with specific topology and compare against Golden."""
        try:
            init_distributed(
                tp_size=1,
                sp_ulysses_size=sp_ulysses_degree,
                sp_ring_size=sp_ring_degree,
            )

            # Init USP model and load golden weights
            usp_decoder = (
                LlamaDecoderLayer(config, attention_backend="usp")
                .to(device)
                .to(torch.bfloat16)
            )
            usp_decoder.load_state_dict(state_dict)

            # Shard data (Split Input)
            extract_func = EXTRACT_FUNC_DICT["basic"]

            local_input_ids = (
                extract_func(
                    data_input_ids,
                    rank,
                    world_size=world_size,
                    rd=sp_ring_degree,
                    ud=sp_ulysses_degree,
                )
                .detach()
                .clone()
            )

            local_hidden_states = (
                extract_func(
                    data_hidden_states,
                    rank,
                    world_size=world_size,
                    rd=sp_ring_degree,
                    ud=sp_ulysses_degree,
                )
                .detach()
                .clone()
            )

            # Run USP forward
            with torch.no_grad():
                usp_output = run_iterative_pass(
                    decoder_layer=usp_decoder,
                    embed_tokens=embed_tokens,
                    input_ids=local_input_ids,
                    hidden_states=local_hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    ttt_length=ttt_length,
                )

            # Verify results
            # Slice the golden output to match the current rank's chunk
            total_degree = sp_ring_degree * sp_ulysses_degree
            chunk_size = sdpa_output.shape[1] // total_degree
            start_idx = (rank % total_degree) * chunk_size
            end_idx = start_idx + chunk_size

            golden_chunk = sdpa_output[:, start_idx:end_idx, :]

            assert torch.allclose(usp_output, golden_chunk, rtol=2e-2, atol=2e-2), (
                f"[Rank {rank}] USP (U{sp_ulysses_degree}R{sp_ring_degree}) mismatch!\n"
                f"Max Diff: {(usp_output - golden_chunk).abs().max().item()}"
            )

        finally:
            destroy_distributed()

    # Case 1: Hybrid (Ulysses=2, Ring=1)
    subtest_usp(sp_ulysses_degree=2, sp_ring_degree=1)

    # Case 2: Hybrid (Ulysses=1, Ring=2)
    subtest_usp(sp_ulysses_degree=1, sp_ring_degree=2)


class TestTTTDistributed(unittest.TestCase):
    def test_llama_usp_decoder(self):
        world_size = 2
        port = get_available_port()
        mp.spawn(run_test_case, nprocs=world_size, args=(world_size, port))


if __name__ == "__main__":
    unittest.main()
