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
from tests.utils import get_available_port, is_port_in_use


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
_GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0


def _model_config() -> PretrainedConfig:
    """Small but kernel-real Llama decoder config for the multi-GPU gates."""
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
        position_ids=torch.arange(local_seq_len, device=device).unsqueeze(0),
        hidden_states=torch.zeros((1, padded_len, 8), device=device),
        target_p_padded=torch.zeros((1, padded_len, 8), device=device),
        position_mask=torch.ones((1, padded_len, 1), device=device),
        seq_length=padded_len,
    )
    assert state.input_ids.shape[1] == local_seq_len
    assert state.hidden_states.shape[1] == local_seq_len
    assert state.position_ids.shape[1] == local_seq_len


def _assert_gradient_close(actual, expected, name):
    """Bound BF16 reduction-order error without hiding small gradients in atol.

    Relative L2 checks the whole tensor; the max-error bound also catches a
    localized mismatch. Both are scaled to the reference, with only a 1e-7
    absolute floor for zero/tiny tensors.
    """
    assert actual is not None and expected is not None, f"missing gradient: {name}"
    actual, expected = actual.float(), expected.float()
    assert torch.isfinite(actual).all(), f"non-finite gradient: {name}"
    error = actual - expected
    relative_l2 = error.norm() / expected.norm().clamp_min(1e-7)
    max_error = error.abs().max()
    assert (
        error.norm() <= 0.02 * expected.norm() + 1e-7
    ), f"{name}: gradient relative L2={relative_l2.item():.6g}"
    assert (
        max_error <= 0.02 * expected.abs().max() + 1e-7
    ), f"{name}: gradient max error={max_error.item():.6g}"
    return relative_l2.item()


def _run_decoder_parity(
    rank: int,
    world_size: int,
    rendezvous_ports: tuple[int, ...],
    topologies: tuple[tuple[int, int], ...],
) -> None:
    device = _setup_worker(rank, world_size, rendezvous_ports[0])
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

    # The embedding is frozen in training. The hidden leaf must have distinct
    # values at every position so token/gradient permutations are observable.
    embed_tokens.requires_grad_(False)
    hidden_states.requires_grad_(True)
    golden_output = _run_iterative_pass(
        golden_decoder,
        embed_tokens,
        global_input_steps,
        hidden_states,
        position_ids,
    )
    # A nonuniform upstream gradient exercises every output position/channel.
    # Each rank uses its slice and the SAME global normalization, so local
    # losses sum to the reference loss (no implicit world-size averaging).
    probe = torch.randn_like(golden_output, dtype=torch.float32) / seq_len
    golden_loss = (golden_output.float() * probe).sum()
    golden_loss.backward()
    golden_hidden_grad = hidden_states.grad.detach().clone()
    golden_grads = {
        name: parameter.grad.detach().float().clone()
        for name, parameter in golden_decoder.named_parameters()
    }
    golden_state_dict = golden_decoder.state_dict()
    golden_output = golden_output.detach()
    golden_loss = golden_loss.detach()
    hidden_states = hidden_states.detach()
    del golden_decoder

    for (sp_ulysses_size, sp_ring_size), rendezvous_port in zip(
        topologies, rendezvous_ports, strict=True
    ):
        try:
            # Each topology creates and destroys a complete default process
            # group. Reusing one TCPStore endpoint is racy, especially when
            # TORCH_DISTRIBUTED_DEBUG=DETAIL adds its Gloo wrapper group.
            os.environ["MASTER_PORT"] = str(rendezvous_port)
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
            # Exercise production position construction, including collator
            # padding and adapter slicing. RoPE consumes local positions
            # before Ulysses exchanges the rotated Q/K.
            from specforge.algorithms.eagle3.data import (
                build_offline_collator,
                build_offline_normalizer,
            )

            normalize = build_offline_normalizer(
                seq_len, ttt_length=ttt_length, use_usp_preprocess=True
            )
            batch = build_offline_collator()(
                [
                    normalize(
                        {
                            "input_ids": input_ids.cpu(),
                            "loss_mask": torch.ones_like(input_ids, device="cpu"),
                            "aux_hidden_state": hidden_states.cpu(),
                            "hidden_state": hidden_states.cpu(),
                        }
                    )
                ]
            )
            view = UspAdapter(object()).backbone_view(
                row_count=seq_len // world_size,
                global_input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"],
                hidden_states=batch["hidden_state"],
            )
            local_position_ids = view.position_ids.to(device)
            local_hidden_states = (
                view.hidden_states.to(device).detach().requires_grad_(True)
            )
            torch.testing.assert_close(
                local_hidden_states,
                _local_sequence_shard(hidden_states, rank, world_size),
                rtol=0,
                atol=0,
            )
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
            local_probe = _local_sequence_shard(probe, rank, world_size)
            local_loss = (usp_output.float() * local_probe).sum()
            local_loss.backward()
            total_loss = local_loss.detach().clone()
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            torch.testing.assert_close(total_loss, golden_loss, rtol=2e-2, atol=2e-2)

            # Communication backward already accumulates contributions for
            # each owned input token. Replicated parameter gradients, however,
            # must be SUM-reduced over the entire SP group before comparison.
            reduced_grads = {}
            for name, parameter in usp_decoder.named_parameters():
                assert parameter.grad is not None, f"missing gradient: {name}"
                grad = parameter.grad.detach().float().clone()
                dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                reduced_grads[name] = grad
            label = f"rank={rank} U{sp_ulysses_size}R{sp_ring_size}"
            hidden_error = _assert_gradient_close(
                local_hidden_states.grad,
                _local_sequence_shard(golden_hidden_grad, rank, world_size),
                f"{label} hidden",
            )
            parameter_error = max(
                _assert_gradient_close(grad, golden_grads[name], f"{label} {name}")
                for name, grad in reduced_grads.items()
            )
            print(
                f"{label}: output max_abs={max_diff:.6g}, "
                f"hidden grad rel_l2={hidden_error:.6g}, "
                f"parameter grad worst_rel_l2={parameter_error:.6g}",
                flush=True,
            )
            # This is a device-less NCCL collective, so identify the GPU
            # explicitly instead of asking NCCL to infer it from global rank.
            dist.barrier(device_ids=[rank])
        finally:
            destroy_distributed()


@unittest.skipUnless(
    _HAS_FLASH_ATTN,
    "standard flash-attn forward/backward and padding interfaces are required",
)
class TestDecoderParity(unittest.TestCase):
    """Real BF16 kernels: three-step outputs, input and parameter gradients."""

    @unittest.skipUnless(_GPU_COUNT >= 2, "requires at least two CUDA devices")
    def test_two_gpu_usp_decoder_matches_flash_attention(self):
        world_size = 2
        first_port = get_available_port()
        second_port = first_port + 1
        while second_port < 65535 and is_port_in_use(second_port):
            second_port += 1
        if second_port == 65535:
            self.fail("could not find a second rendezvous port")
        mp.spawn(
            _run_decoder_parity,
            nprocs=world_size,
            args=(world_size, (first_port, second_port), ((2, 1), (1, 2))),
            join=True,
        )

    @unittest.skipUnless(_GPU_COUNT >= 4, "requires at least four CUDA devices")
    def test_four_gpu_u2r2_decoder_matches_flash_attention(self):
        mp.spawn(
            _run_decoder_parity,
            nprocs=4,
            args=(4, (get_available_port(),), ((2, 2),)),
            join=True,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
