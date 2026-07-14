import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from transformers import PretrainedConfig

from specforge.core.eagle3_adapters import SdpaLikeAdapter, UspAdapter
from specforge.distributed import destroy_distributed, init_distributed
from specforge.layers.ring import ring_flash_attn_func
from specforge.modeling.draft.llama3_eagle import LlamaDecoderLayer
from specforge.utils import padding
from tests.utils import get_available_port


def _standard_flash_attn_available() -> bool:
    """Both the FA golden path and USP require the standard FA interface."""
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
        from flash_attn.bert_padding import pad_input, unpad_input  # noqa: F401
        from flash_attn.flash_attn_interface import (  # noqa: F401
            _flash_attn_varlen_backward,
        )
    except Exception:
        return False
    return True


_HAS_FLASH_ATTN = _standard_flash_attn_available()
_HAS_2_GPUS = torch.cuda.is_available() and torch.cuda.device_count() >= 2


def _model_config() -> PretrainedConfig:
    """Small but kernel-real Llama decoder config for the two-GPU gate."""
    return PretrainedConfig.from_dict(
        {
            "architectures": ["LlamaForCausalLMEagle3"],
            "hidden_act": "silu",
            "hidden_size": 256,
            "intermediate_size": 512,
            "max_position_embeddings": 512,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "num_hidden_layers": 1,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-5,
            "rope_scaling": None,
            "tie_word_embeddings": False,
            "use_cache": True,
            "vocab_size": 1024,
            "draft_vocab_size": 512,
            "pretraining_tp": 1,
        }
    )


def _setup_worker(rank: int, world_size: int, port: int) -> torch.device:
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "SPECFORGE_DEVICE": "cuda",
        }
    )
    torch.cuda.set_device(rank)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    return torch.device("cuda", rank)


def _input_steps(input_ids: torch.Tensor, ttt_length: int) -> list[torch.Tensor]:
    """Build global shifted inputs before sharding to preserve rank boundaries."""
    steps = []
    current = input_ids
    for _ in range(ttt_length):
        steps.append(current)
        current = padding(current, left=False)
    return steps


def _run_iterative_pass(
    decoder_layer: LlamaDecoderLayer,
    embed_tokens: nn.Embedding,
    input_steps: list[torch.Tensor],
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    cache_hidden = [[], []]
    output = hidden_states
    for input_ids in input_steps:
        input_emb = embed_tokens(input_ids).to(output.dtype)
        output = decoder_layer(
            input_emb=input_emb,
            hidden_states=output,
            cache_hidden=cache_hidden,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            output_attentions=False,
            use_cache=False,
        )
    return output


def _local_sequence_shard(tensor: torch.Tensor, rank: int, world_size: int):
    return tensor.chunk(world_size, dim=1)[rank].contiguous()


def _assert_adapter_contract(
    *,
    device: torch.device,
    local_seq_len: int,
    ttt_length: int,
    sp_ulysses_size: int,
) -> None:
    """Exercise the canonical USP adapter with the real process groups."""
    padded_len = local_seq_len + ttt_length
    adapter = UspAdapter(object())
    state = adapter.step_view(
        idx=0,
        ttt_length=ttt_length,
        global_input_ids=torch.zeros((1, padded_len), dtype=torch.long, device=device),
        attention_mask=torch.ones((1, padded_len), device=device),
        loss_mask=torch.ones((1, padded_len, 1), device=device),
        position_ids=torch.arange(
            local_seq_len * sp_ulysses_size, device=device
        ).unsqueeze(0),
        hidden_states=torch.zeros((1, padded_len, 8), device=device),
        target_p_padded=torch.zeros((1, padded_len, 8), device=device),
        position_mask=torch.ones((1, padded_len, 1), device=device),
        seq_length=padded_len,
    )
    assert state.input_ids.shape[1] == local_seq_len
    assert state.hidden_states.shape[1] == local_seq_len
    assert state.position_ids.shape[1] == local_seq_len * sp_ulysses_size


def _run_decoder_parity(rank: int, world_size: int, port: int) -> None:
    device = _setup_worker(rank, world_size, port)
    config = _model_config()
    seq_len = 128
    ttt_length = 3

    input_ids = torch.randint(
        0, config.vocab_size, (1, seq_len), dtype=torch.long, device=device
    )
    hidden_states = torch.randn(
        (1, seq_len, config.hidden_size), dtype=torch.bfloat16, device=device
    )
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    global_input_steps = _input_steps(input_ids, ttt_length)

    embed_tokens = nn.Embedding(
        config.vocab_size, config.hidden_size, config.pad_token_id
    ).to(device=device, dtype=torch.bfloat16)
    golden_decoder = LlamaDecoderLayer(config, attention_backend="fa").to(
        device=device, dtype=torch.bfloat16
    )

    # The non-USP adapter remains the golden path's canonical view contract.
    golden_state = SdpaLikeAdapter(object()).step_view(
        idx=0,
        ttt_length=ttt_length,
        global_input_ids=input_ids,
        attention_mask=torch.ones((1, seq_len), device=device),
        loss_mask=torch.ones((1, seq_len, 1), device=device),
        position_ids=position_ids,
        hidden_states=hidden_states,
        target_p_padded=torch.zeros((1, seq_len, 8), device=device),
        position_mask=torch.ones((1, seq_len, 1), device=device),
        seq_length=seq_len,
    )
    assert golden_state.input_ids.shape[1] == seq_len

    with torch.no_grad():
        golden_output = _run_iterative_pass(
            golden_decoder,
            embed_tokens,
            global_input_steps,
            hidden_states,
            position_ids,
        )
    golden_state_dict = golden_decoder.state_dict()
    del golden_decoder

    for sp_ulysses_size, sp_ring_size in ((2, 1), (1, 2)):
        try:
            init_distributed(
                timeout=2,
                tp_size=1,
                sp_ulysses_size=sp_ulysses_size,
                sp_ring_size=sp_ring_size,
            )
            assert callable(ring_flash_attn_func)
            _assert_adapter_contract(
                device=device,
                local_seq_len=seq_len // world_size,
                ttt_length=ttt_length,
                sp_ulysses_size=sp_ulysses_size,
            )

            usp_decoder = LlamaDecoderLayer(config, attention_backend="usp").to(
                device=device, dtype=torch.bfloat16
            )
            usp_decoder.load_state_dict(golden_state_dict)

            local_input_steps = [
                _local_sequence_shard(step, rank, world_size)
                for step in global_input_steps
            ]
            local_hidden_states = _local_sequence_shard(hidden_states, rank, world_size)
            if sp_ring_size > 1:
                local_position_ids = _local_sequence_shard(
                    position_ids, rank, world_size
                )
            else:
                local_position_ids = position_ids

            with torch.no_grad():
                usp_output = _run_iterative_pass(
                    usp_decoder,
                    embed_tokens,
                    local_input_steps,
                    local_hidden_states,
                    local_position_ids,
                )

            expected = _local_sequence_shard(golden_output, rank, world_size)
            max_diff = (usp_output - expected).abs().max().item()
            assert torch.allclose(usp_output, expected, rtol=2e-2, atol=2e-2), (
                f"rank={rank} USP U{sp_ulysses_size}R{sp_ring_size} decoder "
                f"mismatch; max_diff={max_diff}"
            )
            dist.barrier()
        finally:
            destroy_distributed()


class TestDecoderParity(unittest.TestCase):
    @unittest.skipUnless(
        _HAS_FLASH_ATTN,
        "standard flash-attn forward/backward and padding interfaces are required",
    )
    @unittest.skipUnless(_HAS_2_GPUS, "requires at least two CUDA devices")
    def test_two_gpu_usp_decoder_matches_flash_attention(self):
        world_size = 2
        mp.spawn(
            _run_decoder_parity,
            nprocs=world_size,
            args=(world_size, get_available_port()),
            join=True,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
