import tempfile

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from transformers import Llama4ForCausalLM, Llama4TextConfig

from sgl_spec.distributed import init_distributed
from sgl_spec.modeling.target.llama4 import Llama4ForCausalLM


def test_llama4_tp():
    init_distributed(tp_size=2)
    set_seed(42)
    config = Llama4TextConfig(
        vocab_size=1000,
        hidden_size=384,
        intermediate_size=512,
        intermediate_size_mlp=512,
        num_hidden_layers=1,
        max_position_embeddings=1024,
        num_attention_heads=10,
        num_key_value_heads=2,
        head_dim=64,
        num_local_experts=4,
        tie_word_embedding=False,
        initializer_range=0.02,
        hidden_act="silu",
    )

    # create the single-gpu
    model = Llama4ForCausalLM(config).cuda()

    from sgl_spec.modeling.target.llama4 import (
        Llama4ForCausalLM as DistLlama4ForCausalLM,
    )

    dist_model = DistLlama4ForCausalLM(config).cuda()

    # save the model weights to a temp directory
    if dist.get_rank() == 0:
        temp_dir = tempfile.TemporaryDirectory()
        model.save_pretrained(temp_dir.name)
        print(f"Saved model to {temp_dir.name}")
        temp_path = temp_dir.name
        dist.broadcast_object_list([temp_path], src=0)
    else:
        obj_list = [None]
        dist.broadcast_object_list(obj_list, src=0)
        temp_path = obj_list[0]

    # load the model weights to the distributed model
    print(f"Loading model from {temp_path}")
    dist_model.load_checkpoint(temp_path)
    dist.barrier()

    if dist.get_rank() == 0:
        temp_dir.cleanup()

    # create data
    input_ids = torch.randint(0, 1000, (1, 256)).cuda()
    attention_mask = torch.ones_like(input_ids).cuda()

    expected_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    dist_logits = dist_model(input_ids=input_ids, attention_mask=attention_mask).logits

    assert torch.allclose(
        expected_logits, dist_logits
    ), f"Logits are not close, {expected_logits} vs {dist_logits}"


if __name__ == "__main__":
    test_llama4_tp()
