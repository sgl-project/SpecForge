import os
import pickle
import unittest

import torch
import torch.multiprocessing as mp
from accelerate.utils import set_seed
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import PretrainedConfig
from yunchang import EXTRACT_FUNC_DICT
from yunchang.comm import SeqAllToAll4D

from specforge import AutoEagle3DraftModel, AutoDraftModelConfig, OfflineEagle3Model
from specforge.distributed import init_distributed, get_dp_device_mesh, destroy_distributed, get_sp_ulysses_group
from specforge.modeling.draft.llama3_eagle import LlamaDecoderLayer
from specforge.modeling.target.target_head import TargetHead
from specforge.utils import padding
from tests.utils import get_available_port


def test_usp_decoder_layer(rank, world_size, port):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    sp_ring_degree = 2
    sp_ulysses_degree = 1
    init_distributed(tp_size=1, sp_ulysses_size=4, sp_ring_size=1)
    print(f"rank: {rank}, world_size: {world_size}, port: {port}")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    set_seed(42)
    from transformers import PretrainedConfig

    config = {
        "architectures": [
            "LlamaForCausalLMEagle3"
        ],
        "eagle_config": {
            "eagle_aux_hidden_state_layer_ids": [
                1,
                29,
                57
            ],
            "use_aux_hidden_state": True
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

    # 3. 加载数据并确保在当前GPU上
    input_ids = torch.load("input_ids.pt", map_location=device)  # 加载时指定设备
    input_embeds = embed_tokens(input_ids)  # 自动在GPU上
    hidden_states = torch.load("hidden_states.pt", map_location=device)
    attention_mask = torch.load("attention_mask.pt", map_location=device)
    position_ids = torch.load("position_ids.pt", map_location=device)
    past_key_values = None
    sdpa_decoder_layer = LlamaDecoderLayer(config, attention_backend="sdpa").to(device)
    cache_hidden = [[], []]
    sdpa_output = sdpa_decoder_layer(
        input_emb=input_embeds,
        hidden_states=hidden_states,
        cache_hidden=cache_hidden,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        output_attentions=False,
        use_cache=False,
    )
    # sdpa_output = torch.load(f"sdpa_output_{rank}.pt")
    torch.save(sdpa_output, f"sdpa_output_{rank}.pt")
    destroy_distributed()
    init_distributed(tp_size=1, sp_ulysses_size=1, sp_ring_size=2)

    # torch.save(sdpa_decoder_layer.state_dict(), f"sdpa_decoder_layer_{rank}.pt")
    # create layers
    usp_decoder_layer = LlamaDecoderLayer(config, attention_backend="usp").to(device)
    usp_decoder_layer.load_state_dict(sdpa_decoder_layer.state_dict())
    cache_hidden = [[], []]
    past_key_values = None
    extract_func = EXTRACT_FUNC_DICT["basic"]
    input_ids = (
        extract_func(
            input_ids, rank%(sp_ring_degree*sp_ulysses_degree), world_size=sp_ring_degree*sp_ulysses_degree, rd=sp_ring_degree, ud=sp_ulysses_degree
        )
        .detach()
        .clone()
    )

    hidden_states = (
        extract_func(
            hidden_states, rank%(sp_ring_degree*sp_ulysses_degree), world_size=sp_ring_degree*sp_ulysses_degree, rd=sp_ring_degree, ud=sp_ulysses_degree
        )
        .detach()
        .clone()
    )

    input_embeds = embed_tokens(input_ids)
    usp_output = usp_decoder_layer(
        input_emb=input_embeds,
        hidden_states=hidden_states,
        cache_hidden=cache_hidden,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        output_attentions=False,
        use_cache=False,
    )

    torch.save(usp_output, f"usp_output{torch.distributed.get_rank()}.pt")
    torch.save(sdpa_output, f"sdpa_output{torch.distributed.get_rank()}.pt")

    # check
    assert torch.allclose(
        usp_output, sdpa_output.chunk(2, dim=1)[rank%(sp_ring_degree*sp_ulysses_degree)], rtol=1e-5, atol=1e-5
    ), f"native_output: \n{usp_output}, \nsf_output: \n{sdpa_output}"


def test_all_to_all(rank, world_size, port):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_distributed(tp_size=1, sp_ulysses_size=2, sp_ring_size=2)
    ulysses_pg = get_sp_ulysses_group()
    device = f'cuda:{rank}'
    # bs, seq_len, num_head, hidden_size
    input_tensor = torch.zeros(1, 1, 4, 1).to(device)
    input_tensor = input_tensor + rank
    query_states = SeqAllToAll4D.apply(
        ulysses_pg, input_tensor, 2, 1, True
    )
    print(f"{torch.distributed.get_rank()}. {query_states=}")


def test_ttt(rank, world_size, port):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)


    init_distributed(tp_size=1, sp_ulysses_size=1, sp_ring_size=1)
    ttt_length = 3
    print(f"rank: {rank}, world_size: {world_size}, port: {port}")

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    set_seed(42)
    from transformers import PretrainedConfig

    config = {
        "architectures": [
            "LlamaForCausalLMEagle3"
        ],
        "eagle_config": {
            "eagle_aux_hidden_state_layer_ids": [
                1,
                29,
                57
            ],
            "use_aux_hidden_state": True
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

    # 3. 加载数据并确保在当前GPU上
    input_ids = torch.load("input_ids.pt", map_location=device)  # 加载时指定设备
    hidden_states = torch.load("hidden_states.pt", map_location=device)
    attention_mask = torch.load("attention_mask.pt", map_location=device)
    position_ids = torch.load("position_ids.pt", map_location=device)
    past_key_values = None
    cache_hidden = [[], []]

    sdpa_decoder_layer = LlamaDecoderLayer(config, attention_backend="sdpa").to(device).to(torch.bfloat16)
    for idx in range(ttt_length):
        is_last = idx == ttt_length - 1
        print(f"sdpa {hidden_states.shape=}")

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
    torch.save(sdpa_hidden_states_out, f"sdpa_hidden_states_out_{rank}.pt")
    import pickle
    with open(f"sdpa_cache_hidden_{rank}.pickle", "wb") as f:
        pickle.dump(cache_hidden, f)
    destroy_distributed()

    sp_ulysses_degree = 2
    sp_ring_degree = 2
    init_distributed(tp_size=1, sp_ulysses_size=sp_ulysses_degree, sp_ring_size=sp_ring_degree)

    # sdpa_hidden_states_out = torch.load(f"sdpa_hidden_states_out_{rank}.pt")
    # torch.save(sdpa_decoder_layer.state_dict(), "ttt_sdpa_decoder_layer.pt")
    origin_input_ids = torch.load("input_ids.pt", map_location=device)  # 加载时指定设备
    hidden_states = torch.load("hidden_states.pt", map_location=device)
    attention_mask = torch.load("attention_mask.pt", map_location=device)
    position_ids = torch.load("position_ids.pt", map_location=device)
    usp_decoder_layer = LlamaDecoderLayer(config, attention_backend="usp").to(device).to(torch.bfloat16)
    usp_decoder_layer.load_state_dict(sdpa_decoder_layer.state_dict())
    cache_hidden = [[], []]
    past_key_values = None
    extract_func = EXTRACT_FUNC_DICT["basic"]

    hidden_states = (
        extract_func(
            hidden_states, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
        )
        .detach()
        .clone()
    )

    for idx in range(ttt_length):
        input_ids = (
            extract_func(
                origin_input_ids, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
            )
            .detach()
            .clone()
        )
        is_last = idx == ttt_length - 1
        print(f"usp {hidden_states.shape=}, {hidden_states.dtype=}")
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
    with open(f"usp_cache_hidden_{rank}.pickle", "wb") as f:
        pickle.dump(cache_hidden, f)

    torch.save(usp_hidden_states_out, f"usp_hidden_states_out_{rank}.pt")
    torch.save(sdpa_hidden_states_out, f"sdpa_hidden_states_out_{rank}.pt")

    # check, bf16 has larger differ. set threshold to 2e-2
    assert torch.allclose(
        usp_hidden_states_out, sdpa_hidden_states_out.chunk(sp_ring_degree * sp_ulysses_degree, dim=1)[rank%(sp_ring_degree * sp_ulysses_degree)], rtol=2e-2, atol=2e-2
    ), f"usp_output: \n{usp_hidden_states_out}, \nsdpa_output: \n{sdpa_hidden_states_out}"


def test_ttt_backward(rank, world_size, port):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    sp_ring_degree = 1
    sp_ulysses_degree = 1
    init_distributed(tp_size=1, sp_ulysses_size=sp_ulysses_degree, sp_ring_size=sp_ring_degree)
    args = {
        "draft_model_config": "./configs/deepseek-r1-671b-eagle3.json",
        "ttt_length": 3,
        "target_model_path": "/nfs/ofs-llab-cold/model/deepseek-ai/DeepSeek-V3.1",
        "draft_attention_backend": "sdpa",
    }
    args = PretrainedConfig.from_dict(args)

    draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)
    vocab_mapping_path = "./cache/vocab_mapping/bfff91f09e60b28fe9bbf5470c2cddaf.pt"
    draft_model = (
        AutoEagle3DraftModel.from_config(
            draft_model_config, attention_backend=args.draft_attention_backend
        )
        .cuda()
        .to(torch.bfloat16)
    )
    draft_model.load_vocab_mapping(vocab_mapping_path)
    draft_model_state_dict = draft_model.state_dict()
    with open("data_rank_0.pkl", "rb") as f:
        data = pickle.load(f)

    target_head = TargetHead(args.target_model_path)
    target_head = target_head.eval().cuda().to(torch.bfloat16)
    target_head_state_dict = target_head.state_dict()
    # copy for usp model

    ###

    sdpa_model = OfflineEagle3Model(
        target_head=target_head,
        draft_model=draft_model,
        length=args.ttt_length,
        attention_backend=args.draft_attention_backend,
    )
    sdpa_model_state_dict = sdpa_model.state_dict()
    ##
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    fsdp_config = {"mesh": get_dp_device_mesh(), "mp_policy": mp_policy}
    fully_shard(sdpa_model, **fsdp_config)


    plosses, _, acces = sdpa_model(
        input_ids=data["input_ids"].cuda(),  # [B, S]
        attention_mask=data["attention_mask"].cuda(),  # [B, S]
        loss_mask=data["loss_mask"]
        .unsqueeze(-1)
        .cuda(),  # [B, S, 1] This is different from the online version
        hidden_states=data["hidden_state"].cuda(),  # [B, S, D]
        target=data["target"].cuda(),  # [B, S, D*3]
    )
    # calculate weighted loss
    ploss_weight = [0.8 ** i for i in range(len(plosses))]
    args.draft_accumulation_steps = 1
    ploss = (
        sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
        / args.draft_accumulation_steps
    )
    ploss.backward()
    torch.save(sdpa_model.draft_model.midlayer.self_attn.o_proj.weight.grad, f"sdpa_o_proj_grad_{torch.distributed.get_rank()}.pt", )
    destroy_distributed()


    # =================
    init_distributed(tp_size=1, sp_ulysses_size=2, sp_ring_size=sp_ring_degree)

    args = {
        "draft_model_config": "./configs/deepseek-r1-671b-eagle3.json",
        "ttt_length": 3,
        "target_model_path": "/nfs/ofs-llab-cold/model/deepseek-ai/DeepSeek-V3.1",
        "draft_attention_backend": "usp",
    }
    args = PretrainedConfig.from_dict(args)
    usp_draft_model = (
        AutoEagle3DraftModel.from_config(
            draft_model_config, attention_backend="usp"
        )
        .cuda()
        .to(torch.bfloat16)
    )
    usp_draft_model.load_vocab_mapping(vocab_mapping_path)
    usp_draft_model.load_state_dict(draft_model_state_dict)
    usp_target_head = TargetHead(args.target_model_path)
    usp_target_head.load_state_dict(target_head_state_dict)
    usp_target_head = usp_target_head.eval().cuda().to(torch.bfloat16)

    usp_model = OfflineEagle3Model(
        target_head=usp_target_head,
        draft_model=usp_draft_model,
        length=args.ttt_length,
        attention_backend="usp",
    )
    usp_model.load_state_dict(sdpa_model_state_dict)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    fsdp_config = {"mesh": get_dp_device_mesh(), "mp_policy": mp_policy}
    fully_shard(usp_model, **fsdp_config)

    plosses, _, acces = usp_model(
        input_ids=data["input_ids"].cuda(),  # [B, S]
        attention_mask=data["attention_mask"].cuda(),  # [B, S]
        loss_mask=data["loss_mask"]
        .unsqueeze(-1)
        .cuda(),  # [B, S, 1] This is different from the online version
        hidden_states=data["hidden_state"].cuda(),  # [B, S, D]
        target=data["target"].cuda(),  # [B, S, D*3]
    )
    # calculate weighted loss
    ploss_weight = [0.8 ** i for i in range(len(plosses))]
    args.draft_accumulation_steps = 1
    ploss = (
        sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
        / args.draft_accumulation_steps
    )

    ploss.backward()
    torch.save( usp_model.draft_model.midlayer.self_attn.o_proj.weight.grad, f"usp_o_proj_grad_{torch.distributed.get_rank()}.pt",)


class TestLinear(unittest.TestCase):
    def test_usp_decoder_layer1(self):
        port = get_available_port()
        mp.spawn(test_usp_decoder_layer, nprocs=4, args=(4, port))


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestLinear))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
