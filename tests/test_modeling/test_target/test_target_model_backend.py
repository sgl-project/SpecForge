import os
import unittest

import torch
import torch.multiprocessing as mp
from accelerate.utils import set_seed

from specforge.distributed import init_distributed
from specforge.modeling.target.eagle3_target_model import (
    CustomEagle3TargetModel,
    HFEagle3TargetModel,
    SGLangEagle3TargetModel,
)
from tests.utils import get_available_port


@torch.no_grad()
def test_target_model_backend(rank, world_size, port, tp_size):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    init_distributed(tp_size=tp_size)
    set_seed(42)

    input_ids = torch.randint(0, 1000, (2, 256)).cuda()
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.ones_like(input_ids)

    hf_target_model = HFEagle3TargetModel.from_pretrained(
        "unsloth/Llama-3.2-1B", torch_dtype=torch.float16, device="cuda"
    )
    hf_target_model.set_aux_hidden_states_layers()
    hf_out = hf_target_model.generate_eagle3_data(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
    )
    del hf_target_model

    custom_target_model = CustomEagle3TargetModel.from_pretrained(
        "unsloth/Llama-3.2-1B", torch_dtype=torch.float16, device="cuda"
    )
    custom_target_model.set_aux_hidden_states_layers()
    custom_out = custom_target_model.generate_eagle3_data(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
    )
    del custom_target_model

    # compare weights
    assert torch.allclose(
        hf_out.target, custom_out.target, atol=1e-5, rtol=1e-5
    ), f"Logits are not close: \nhf: {hf_out[0] - custom_out[0]}"
    assert torch.allclose(
        hf_out.loss_mask, custom_out.loss_mask, atol=1e-5, rtol=1e-5
    ), f"Logits are not close: \ndiff: {hf_out[1] - custom_out[1]}"
    assert torch.allclose(
        hf_out.input_ids, custom_out.input_ids, atol=1e-5, rtol=1e-5
    ), f"Logits are not close: \ndiff: {hf_out[1] - custom_out[1]}"
    assert torch.allclose(
        hf_out.hidden_states, custom_out.hidden_states, atol=1e-5, rtol=1e-5
    ), f"Logits are not close: \ndiff: {hf_out[1] - custom_out[1]}"

    sgl_target_model = SGLangEagle3TargetModel.from_pretrained(
        "unsloth/Llama-3.2-1B", torch_dtype=torch.float16, device="cuda"
    )
    sgl_target_model.set_aux_hidden_states_layers()
    sgl_out = sgl_target_model.generate_eagle3_data(
        input_ids=input_ids, attention_mask=attention_mask, loss_mask=loss_mask
    )
    del sgl_target_model

    assert torch.equal(hf_out.loss_mask, sgl_out.loss_mask)
    assert torch.equal(hf_out.input_ids, sgl_out.input_ids)
    assert torch.allclose(
        hf_out.hidden_states, sgl_out.hidden_states, atol=1e-1, rtol=1e-2
    ), f"Hidden states are not close, diff: \n{(hf_out.hidden_states - sgl_out.hidden_states).abs().max()}"
    assert torch.allclose(
        hf_out.target, sgl_out.target.half(), atol=1e-1, rtol=1e-2
    ), f"Target are not close, diff: \n{(hf_out.target - sgl_out.target).abs().max()}"


class TestTargetModelBackend(unittest.TestCase):

    def test_target_model_backend_dp(self):
        world_size = 2
        port = get_available_port()
        mp.spawn(
            test_target_model_backend, nprocs=world_size, args=(world_size, port, 1)
        )

    def test_target_model_backend_tp(self):
        world_size = 2
        port = get_available_port()
        mp.spawn(
            test_target_model_backend, nprocs=world_size, args=(world_size, port, 2)
        )


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestTargetModelBackend))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
