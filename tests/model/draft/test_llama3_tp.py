# Filename: test_tp_correctness.py (Final version with tests for both MLP and Attention)

import os
import tempfile
import unittest

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from transformers import LlamaConfig
from transformers.activations import ACT2FN

from specforge.distributed import destroy_distributed, init_distributed
from specforge.model.draft.llama3_eagle import LlamaAttention, LlamaMLP


# === Temporary, Non-Parallel Model Definitions (for this test file only) ===
class VanillaLlamaMLP(nn.Module):
    """Temporary non-parallel model to generate the MLP baseline answer."""

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# To make VanillaLlamaAttention work standalone, we need to copy some helper functions and classes.
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class VanillaLlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )

    def forward(self, x, seq_len=None):
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )


class VanillaLlamaAttention(nn.Module):
    """Temporary non-parallel model to generate the Attention baseline answer."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # The input for Eagle Attention is hidden_size * 2
        self.q_proj = nn.Linear(
            self.hidden_size * 2, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self.rotary_emb = VanillaLlamaRotaryEmbedding(self.head_dim)

    def forward(self, hidden_states, position_ids):
        bsz, q_len, _ = hidden_states.size()
        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        cos, sin = self.rotary_emb(query_states, seq_len=q_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, is_causal=True
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)


# === Core Parallel Test Functions ===


def run_mlp_tp_test(rank, world_size, temp_dir_name):
    """
    This function executes the parallel computation for MLP and compares it with the 'golden standard'.
    Args:
        rank (int): The rank of the current process, passed automatically by mp.spawn.
        world_size (int): The total number of processes, passed automatically by mp.spawn.
        temp_dir_name (str): The path to the temporary directory for saving and loading weights.
    """
    os.environ["RANK"], os.environ["WORLD_SIZE"] = str(rank), str(world_size)
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "localhost", "29503"
    init_distributed(timeout=10, target_tp_size=1, draft_tp_size=world_size)
    torch.cuda.set_device(rank)
    config = LlamaConfig(
        hidden_size=128,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=512,
    )

    # Load the actual LlamaMLP with parallel layers from your project
    mlp_tp2 = LlamaMLP(config).cuda(rank)
    full_state_dict = torch.load(os.path.join(temp_dir_name, "mlp_weights.pth"))

    sharded_state_dict = {}
    for name, param in full_state_dict.items():
        if "gate_proj.weight" in name or "up_proj.weight" in name:
            sharded_param = param.chunk(world_size, dim=0)[rank]
        elif "down_proj.weight" in name:
            sharded_param = param.chunk(world_size, dim=1)[rank]
        else:
            sharded_param = param
        sharded_state_dict[name] = sharded_param
    mlp_tp2.load_state_dict(sharded_state_dict)
    mlp_tp2.eval()

    input_tensor = torch.load(os.path.join(temp_dir_name, "mlp_input.pth")).cuda(rank)
    output_tp2 = mlp_tp2(input_tensor)

    if rank == 0:
        output_tp1 = torch.load(os.path.join(temp_dir_name, "mlp_output.pth"))
        assert torch.allclose(
            output_tp1, output_tp2.cpu(), rtol=1e-4, atol=1e-5
        ), "Output mismatch for LlamaMLP between TP=1 and TP=2!"
        print("✅ LlamaMLP TP correctness test passed!")
    destroy_distributed()


def run_attention_tp_test(rank, world_size, temp_dir_name):
    """This function executes the parallel computation for Attention and compares it with the 'golden standard'."""
    os.environ["RANK"], os.environ["WORLD_SIZE"] = str(rank), str(world_size)
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "localhost", "29504"
    init_distributed(draft_tp_size=world_size)
    torch.cuda.set_device(rank)
    config = LlamaConfig(
        hidden_size=128,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=512,
    )

    # Load the actual LlamaAttention with parallel layers from your project
    attn_tp2 = LlamaAttention(config).cuda(rank)
    full_state_dict = torch.load(os.path.join(temp_dir_name, "attn_weights.pth"))

    sharded_state_dict = {}
    for name, param in full_state_dict.items():
        if "rotary_emb" in name:
            sharded_param = param
        elif any(
            s in name for s in ["q_proj.weight", "k_proj.weight", "v_proj.weight"]
        ):
            sharded_param = param.chunk(world_size, dim=0)[rank]
        elif "o_proj.weight" in name:
            sharded_param = param.chunk(world_size, dim=1)[rank]
        else:
            sharded_param = param
        sharded_state_dict[name] = sharded_param
    attn_tp2.load_state_dict(sharded_state_dict, strict=False)
    attn_tp2.eval()

    input_tensor = torch.load(os.path.join(temp_dir_name, "attn_input.pth")).cuda(rank)
    pos_ids = torch.load(os.path.join(temp_dir_name, "attn_pos_ids.pth")).cuda(rank)
    output_tp2 = attn_tp2(input_tensor, position_ids=pos_ids)

    if rank == 0:
        output_tp1 = torch.load(os.path.join(temp_dir_name, "attn_output.pth"))
        assert torch.allclose(
            output_tp1, output_tp2.cpu(), rtol=1e-4, atol=1e-5
        ), "Output mismatch for LlamaAttention between TP=1 and TP=2!"
        print("✅ LlamaAttention TP correctness test passed!")
    destroy_distributed()


# === unittest Launcher ===
class TestTPCorrectness(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_mlp_correctness(self):
        world_size = 2
        temp_dir_path = self.temp_dir.name
        print("\n--- Running MLP TP Correctness Test ---")

        # Phase 1: Generate the "golden standard" for MLP
        torch.manual_seed(42)
        config = LlamaConfig(
            hidden_size=128,
            num_attention_heads=8,
            num_key_value_heads=4,
            intermediate_size=512,
        )
        mlp_tp1 = VanillaLlamaMLP(config)
        mlp_tp1.eval()
        input_tensor = torch.randn(2, 10, config.hidden_size)
        output_tp1 = mlp_tp1(input_tensor)

        torch.save(mlp_tp1.state_dict(), os.path.join(temp_dir_path, "mlp_weights.pth"))
        torch.save(input_tensor, os.path.join(temp_dir_path, "mlp_input.pth"))
        torch.save(output_tp1, os.path.join(temp_dir_path, "mlp_output.pth"))

        # Phase 2 & 3: Spawn parallel processes
        mp.spawn(run_mlp_tp_test, nprocs=world_size, args=(world_size, temp_dir_path))

    def test_attention_correctness(self):
        world_size = 2
        temp_dir_path = self.temp_dir.name
        print("\n--- Running Attention TP Correctness Test ---")

        # Phase 1: Generate the "golden standard" for Attention
        torch.manual_seed(42)
        config = LlamaConfig(
            hidden_size=128,
            num_attention_heads=8,
            num_key_value_heads=4,
            intermediate_size=512,
        )
        attn_tp1 = VanillaLlamaAttention(config)
        attn_tp1.eval()

        input_tensor = torch.randn(2, 10, config.hidden_size * 2)
        pos_ids = torch.arange(10, dtype=torch.long).unsqueeze(0).expand(2, -1)
        output_tp1 = attn_tp1(input_tensor, position_ids=pos_ids)

        torch.save(
            attn_tp1.state_dict(), os.path.join(temp_dir_path, "attn_weights.pth")
        )
        torch.save(input_tensor, os.path.join(temp_dir_path, "attn_input.pth"))
        torch.save(pos_ids, os.path.join(temp_dir_path, "attn_pos_ids.pth"))
        torch.save(output_tp1, os.path.join(temp_dir_path, "attn_output.pth"))

        # Phase 2 & 3: Spawn parallel processes
        mp.spawn(
            run_attention_tp_test, nprocs=world_size, args=(world_size, temp_dir_path)
        )


if __name__ == "__main__":
    unittest.main()
