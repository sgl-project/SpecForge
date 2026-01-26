import os
import time
import unittest

import torch
import torch.multiprocessing as mp
from accelerate.utils import set_seed
from torch import nn
from transformers import PretrainedConfig
from yunchang import EXTRACT_FUNC_DICT

from specforge.core.eagle3_adapters import SdpaLikeAdapter, UspAdapter
from specforge.data.preprocessing import build_offline_eagle3_dataset

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


def dbg(rank, msg):
    print(f"[rank{rank}] {msg}", flush=True)


def wait_for_file(path, timeout_s=60, poll_s=0.1):
    start = time.time()
    while time.time() - start < timeout_s:
        if os.path.exists(path):
            return True
        time.sleep(poll_s)
    return False


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
    dbg(rank, "env setup complete")

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

    # --- Phase 1: Golden Run (FA) ---
    # Init dist briefly for internal checks, even if running single-device logic
    init_distributed(tp_size=1, sp_ulysses_size=1, sp_ring_size=1)
    dbg(rank, "init_distributed (FA) done")

    sdpa_decoder = (
        LlamaDecoderLayer(config, attention_backend="fa").to(device).to(torch.bfloat16)
    )
    dbg(rank, "FA decoder created")
    # Adapter smoke test for FA/SDPA-style path
    dummy_model = type("Dummy", (), {})()
    sdpa_adapter = SdpaLikeAdapter(dummy_model)
    sdpa_target_p = torch.zeros((1, seq_len, 8), device=device, dtype=torch.float32)
    sdpa_position_mask = torch.ones((1, seq_len, 1), device=device, dtype=torch.float32)
    sdpa_state = sdpa_adapter.step_view(
        idx=0,
        ttt_length=ttt_length,
        global_input_ids=data_input_ids,
        attention_mask=attention_mask,
        loss_mask=torch.ones((1, seq_len, 1), device=device, dtype=torch.float32),
        position_ids=position_ids,
        hidden_states=data_hidden_states,
        target_p_padded=sdpa_target_p,
        position_mask=sdpa_position_mask,
        seq_length=seq_len,
    )
    assert sdpa_state.input_ids.shape[1] == seq_len
    assert sdpa_state.hidden_states.shape[1] == seq_len

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
    dbg(rank, "FA forward done")

    # Save weights for alignment and cleanup SDPA model
    state_dict = sdpa_decoder.state_dict()
    del sdpa_decoder
    destroy_distributed()
    dbg(rank, "destroy_distributed (FA) done")

    # --- Phase 2: Distributed Run (USP) ---
    def subtest_usp(sp_ulysses_degree, sp_ring_degree):
        """Run USP with specific topology and compare against Golden."""
        try:
            init_distributed(
                tp_size=1,
                sp_ulysses_size=sp_ulysses_degree,
                sp_ring_size=sp_ring_degree,
            )
            dbg(
                rank,
                f"init_distributed (USP U{sp_ulysses_degree} R{sp_ring_degree}) done",
            )
            # Dataset + adapter smoke test (USP path)
            tmp_dir = "./tmp/usp_dataset_shared"
            try:
                if rank == 0:
                    os.makedirs(tmp_dir, exist_ok=True)
                    sample = {
                        "input_ids": data_input_ids[0].cpu(),
                        "loss_mask": torch.ones_like(data_input_ids[0].cpu()),
                        "hidden_state": data_hidden_states[0].cpu().unsqueeze(0),
                        "aux_hidden_state": data_hidden_states[0].cpu().unsqueeze(0),
                    }
                    torch.save(sample, os.path.join(tmp_dir, "data_0.ckpt"))
                    dbg(rank, "wrote sample ckpt")
                    ready_flag = os.path.join(tmp_dir, "ready.flag")
                    with open(ready_flag, "w", encoding="utf-8") as f:
                        f.write("ready\n")
                if rank != 0:
                    ready_flag = os.path.join(tmp_dir, "ready.flag")
                    assert wait_for_file(
                        ready_flag, timeout_s=60
                    ), "timeout waiting for ready flag"
                dbg(rank, "dataset sync done")
                assert os.path.exists(
                    os.path.join(tmp_dir, "data_0.ckpt")
                ), f"Expected sample not found at {tmp_dir}"
                dbg(rank, "sample exists")

                ds = build_offline_eagle3_dataset(
                    tmp_dir,
                    max_len=seq_len,
                    ttt_length=ttt_length,
                    use_usp_preprocess=True,
                )
                dbg(rank, "dataset built")
                item = ds[0]
                dbg(rank, "dataset item loaded")
                assert "position_ids" in item

                dummy_model = type("Dummy", (), {})()
                adapter = UspAdapter(dummy_model)
                local_seq_len = item["input_ids"].shape[1]
                target_p_padded = torch.zeros(
                    (1, local_seq_len, 8), device=device, dtype=torch.float32
                )
                position_mask = torch.ones(
                    (1, local_seq_len, 1), device=device, dtype=torch.float32
                )
                state = adapter.step_view(
                    idx=0,
                    ttt_length=ttt_length,
                    global_input_ids=item["input_ids"].to(device),
                    attention_mask=item["attention_mask"].to(device),
                    loss_mask=item["loss_mask"].to(device).unsqueeze(-1),
                    position_ids=item["position_ids"].to(device),
                    hidden_states=item["hidden_state"].to(device),
                    target_p_padded=target_p_padded,
                    position_mask=position_mask,
                    seq_length=local_seq_len,
                )
                assert state.input_ids.shape[1] == local_seq_len - ttt_length
                assert state.hidden_states.shape[1] == local_seq_len - ttt_length
                dbg(rank, "adapter step_view ok")
            finally:
                if rank == 0:
                    done_flag = os.path.join(tmp_dir, "done.flag")
                    assert wait_for_file(
                        done_flag, timeout_s=60
                    ), "timeout waiting for done flag"
                    try:
                        for root, _, files in os.walk(tmp_dir):
                            for name in files:
                                os.remove(os.path.join(root, name))
                        os.rmdir(tmp_dir)
                    except OSError:
                        pass
                else:
                    done_flag = os.path.join(tmp_dir, "done.flag")
                    with open(done_flag, "w", encoding="utf-8") as f:
                        f.write("done\n")

            # Init USP model and load golden weights
            usp_decoder = (
                LlamaDecoderLayer(config, attention_backend="usp")
                .to(device)
                .to(torch.bfloat16)
            )
            usp_decoder.load_state_dict(state_dict)
            dbg(rank, "USP decoder loaded")

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
            dbg(rank, "USP local inputs prepared")
            total_degree = sp_ring_degree * sp_ulysses_degree
            chunk_size = sdpa_output.shape[1] // total_degree
            start_idx = (rank % total_degree) * chunk_size
            local_len = local_input_ids.shape[1]
            local_position_ids = (
                torch.arange(start_idx, start_idx + local_len, device=device)
                .unsqueeze(0)
                .long()
            )
            local_attention_mask = torch.tril(
                torch.ones(local_len, local_len, device=device)
            ).view(1, 1, local_len, local_len)

            # Run USP forward
            if sp_ring_degree > 1:
                usp_attention_mask = local_attention_mask
                usp_position_ids = local_position_ids
            else:
                usp_attention_mask = attention_mask
                usp_position_ids = position_ids
            with torch.no_grad():
                usp_output = run_iterative_pass(
                    decoder_layer=usp_decoder,
                    embed_tokens=embed_tokens,
                    input_ids=local_input_ids,
                    hidden_states=local_hidden_states,
                    attention_mask=usp_attention_mask,
                    position_ids=usp_position_ids,
                    ttt_length=ttt_length,
                )
            dbg(rank, "USP forward done")

            # Verify results
            # Slice the golden output to match the current rank's chunk
            end_idx = start_idx + chunk_size

            golden_chunk = sdpa_output[:, start_idx:end_idx, :]

            assert torch.allclose(usp_output, golden_chunk, rtol=2e-2, atol=2e-2), (
                f"[Rank {rank}] USP (U{sp_ulysses_degree}R{sp_ring_degree}) mismatch!\n"
                f"Max Diff: {(usp_output - golden_chunk).abs().max().item()}"
            )
            dbg(rank, "USP output verified")

        finally:
            destroy_distributed()
            dbg(rank, "destroy_distributed (USP) done")

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
