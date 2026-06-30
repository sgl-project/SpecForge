import gc
import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate.utils import set_seed

from specforge.distributed import init_distributed
from specforge.modeling.target.eagle3_target_model import (
    CustomEagle3TargetModel,
    HFEagle3TargetModel,
    SGLangEagle3TargetModel,
)
from tests.utils import get_available_port


def _silence_sglang_allreduce_finalizer():
    """Disable SGLang's custom all-reduce communicators before the worker exits.

    Their ``__del__`` otherwise runs at interpreter shutdown, by which point the
    ``torch`` module global is already ``None``, so ``free()`` logs an ignored
    ``AttributeError: 'NoneType' object has no attribute 'distributed'``. Marking
    them disabled makes the finalizer a no-op; the OS reclaims the buffers on
    process exit. Reaching into SGLang internals is acceptable for test teardown.
    """
    try:
        from sglang.srt.distributed import parallel_state as ps
    except Exception:
        return
    for name in ("_TP", "_WORLD", "_PP", "_MOE_EP", "_MOE_TP", "_ATTN_TP", "_ATTN_CP"):
        group = getattr(ps, name, None)
        ca_comm = getattr(group, "ca_comm", None) if group is not None else None
        if ca_comm is not None:
            ca_comm.disabled = True


def cleanup_distributed():
    _silence_sglang_allreduce_finalizer()
    gc.collect()
    torch.cuda.empty_cache()
    if dist.is_available() and dist.is_initialized():
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
        try:
            dist.destroy_process_group()
        except RuntimeError:
            pass


@torch.no_grad()
def test_target_model_backend(rank, world_size, port, tp_size):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    input_ids = attention_mask = loss_mask = None
    hf_target_model = custom_target_model = sgl_target_model = None
    hf_out = custom_out = sgl_out = None
    try:
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
        hf_target_model = None

        custom_target_model = CustomEagle3TargetModel.from_pretrained(
            "unsloth/Llama-3.2-1B", torch_dtype=torch.float16, device="cuda"
        )
        custom_target_model.set_aux_hidden_states_layers()
        custom_out = custom_target_model.generate_eagle3_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )
        custom_target_model = None

        # Compare the custom backend against the HF reference. The integer/boolean
        # fields (input_ids, loss_mask) must match exactly. The floating point
        # fields (target logits, hidden states) match exactly only without tensor
        # parallelism; with tp_size > 1 the tensor-parallel all-reduce reorders the
        # fp16 accumulation, so a looser tolerance (matching the sglang comparison
        # below) is used.
        assert torch.equal(
            hf_out.input_ids, custom_out.input_ids
        ), f"input_ids differ:\ndiff: {(hf_out.input_ids - custom_out.input_ids).abs().max()}"
        assert torch.equal(
            hf_out.loss_mask, custom_out.loss_mask
        ), f"loss_mask differ:\ndiff: {(hf_out.loss_mask - custom_out.loss_mask).abs().max()}"

        float_atol, float_rtol = (1e-5, 1e-5) if tp_size == 1 else (1e-1, 1e-2)
        assert torch.allclose(
            hf_out.target, custom_out.target, atol=float_atol, rtol=float_rtol
        ), f"Target are not close:\ndiff: {(hf_out.target - custom_out.target).abs().max()}"
        assert torch.allclose(
            hf_out.hidden_states,
            custom_out.hidden_states,
            atol=float_atol,
            rtol=float_rtol,
        ), f"Hidden states are not close:\ndiff: {(hf_out.hidden_states - custom_out.hidden_states).abs().max()}"

        sgl_target_model = SGLangEagle3TargetModel.from_pretrained(
            "unsloth/Llama-3.2-1B", torch_dtype=torch.float16, device="cuda"
        )
        sgl_target_model.set_aux_hidden_states_layers()
        sgl_out = sgl_target_model.generate_eagle3_data(
            input_ids=input_ids, attention_mask=attention_mask, loss_mask=loss_mask
        )
        sgl_target_model = None

        assert torch.equal(hf_out.loss_mask, sgl_out.loss_mask)
        assert torch.equal(hf_out.input_ids, sgl_out.input_ids)
        assert torch.allclose(
            hf_out.hidden_states, sgl_out.hidden_states, atol=1e-1, rtol=1e-2
        ), f"Hidden states are not close, diff: \n{(hf_out.hidden_states - sgl_out.hidden_states).abs().max()}"
        assert torch.allclose(
            hf_out.target, sgl_out.target.half(), atol=1e-1, rtol=1e-2
        ), f"Target are not close, diff: \n{(hf_out.target - sgl_out.target).abs().max()}"
    finally:
        del (
            sgl_out,
            sgl_target_model,
            custom_out,
            custom_target_model,
            hf_out,
            hf_target_model,
            input_ids,
            attention_mask,
            loss_mask,
        )
        cleanup_distributed()


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
