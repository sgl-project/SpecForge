import os
import unittest

import torch
import torch.multiprocessing as mp
from accelerate.utils import set_seed
from torch import nn
from yunchang import EXTRACT_FUNC_DICT

from specforge.distributed import destroy_distributed, init_distributed
from specforge.modeling.draft.llama3_eagle import LlamaDecoderLayer
from specforge.utils import padding
from tests.utils import get_available_port


def test_ttt(rank, world_size, port):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    init_distributed(tp_size=1, sp_ulysses_size=1, sp_ring_size=1)
    ttt_length = 3
    seq_len = 1560
    batch_size = 1
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    set_seed(42)
    from transformers import PretrainedConfig

    config = {
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
    config = PretrainedConfig.from_dict(config)
    # ===============================
    # Case 1: normal layout
    # ===============================
    # create data
    embed_tokens = nn.Embedding(
        config.vocab_size, config.hidden_size, config.pad_token_id
    ).to(device)

    # 3.
    data_input_ids = torch.randint(0, 10000, (batch_size, seq_len), device="cuda")
    data_hidden_states = (
        torch.randn(batch_size, seq_len, config.hidden_size).cuda().to(torch.bfloat16)
    )
    attention_mask = (
        torch.tril(torch.ones(seq_len, seq_len, device=device))
        .view(1, 1, seq_len, seq_len)
        .cuda()
    )

    position_ids = torch.arange(seq_len).unsqueeze(0).cuda()
    hidden_states = data_hidden_states.clone()
    input_ids = data_input_ids.clone()

    past_key_values = None
    cache_hidden = [[], []]

    sdpa_decoder_layer = (
        LlamaDecoderLayer(config, attention_backend="sdpa")
        .to(device)
        .to(torch.bfloat16)
    )
    for idx in range(ttt_length):
        is_last = idx == ttt_length - 1

        # Step 5.1: embed the input ids
        inputs_embeds = embed_tokens(input_ids)
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)

        # Step 5.2: run the draft model backbone
        sdpa_hidden_states_out = sdpa_decoder_layer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=False,
            use_cache=False,
        )

        # update hidden states for next step
        hidden_states = sdpa_hidden_states_out

        if not is_last:
            # Step 5.7: we need to update the loss mask
            input_ids = padding(input_ids, left=False)
    destroy_distributed()

    sp_ulysses_degree = 2
    sp_ring_degree = 1
    init_distributed(
        tp_size=1, sp_ulysses_size=sp_ulysses_degree, sp_ring_size=sp_ring_degree
    )
    origin_input_ids = data_input_ids.clone()
    hidden_states = data_hidden_states.clone()
    usp_decoder_layer = (
        LlamaDecoderLayer(config, attention_backend="usp").to(device).to(torch.bfloat16)
    )
    usp_decoder_layer.load_state_dict(sdpa_decoder_layer.state_dict())
    cache_hidden = [[], []]
    past_key_values = None
    extract_func = EXTRACT_FUNC_DICT["basic"]

    hidden_states = (
        extract_func(
            hidden_states,
            rank,
            world_size=world_size,
            rd=sp_ring_degree,
            ud=sp_ulysses_degree,
        )
        .detach()
        .clone()
    )

    for idx in range(ttt_length):
        input_ids = (
            extract_func(
                origin_input_ids,
                rank,
                world_size=world_size,
                rd=sp_ring_degree,
                ud=sp_ulysses_degree,
            )
            .detach()
            .clone()
        )
        is_last = idx == ttt_length - 1
        # Step 5.1: embed the input ids
        inputs_embeds = embed_tokens(input_ids)
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)

        # Step 5.2: run the draft model backbone
        usp_hidden_states_out = usp_decoder_layer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=False,
            use_cache=False,
        )

        # update hidden states for next step
        hidden_states = usp_hidden_states_out

        if not is_last:
            # Step 5.7: we need to update the loss mask
            origin_input_ids = padding(origin_input_ids, left=False)
    # check, bf16 has larger differ. set threshold to 2e-2
    assert torch.allclose(
        usp_hidden_states_out,
        sdpa_hidden_states_out.chunk(sp_ring_degree * sp_ulysses_degree, dim=1)[
            rank % (sp_ring_degree * sp_ulysses_degree)
        ],
        rtol=2e-2,
        atol=2e-2,
    ), f"usp_output: \n{usp_hidden_states_out}, \nsdpa_output: \n{sdpa_hidden_states_out}"


class TestLinear(unittest.TestCase):
    def test_usp_decoder_layer1(self):
        port = get_available_port()
        mp.spawn(test_ttt, nprocs=2, args=(2, port))


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestLinear))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
