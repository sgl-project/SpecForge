import gc
import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate.utils import set_seed

from specforge.distributed import init_distributed
from specforge.inference.media import MediaInputs
from specforge.inference.target_engine import SGLangTargetEngine
from tests.utils import get_available_port


def _silence_sglang_allreduce_finalizer():
    """Disable SGLang communicators before interpreter teardown."""
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
def _run_capture(rank, world_size, port, tp_size, model_path, **load_kwargs):
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
    target = SGLangTargetEngine.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device="cuda",
        attention_backend="fa3",
        mem_fraction_static=0.4,
        **load_kwargs,
    )
    target.set_capture_layers()
    output = target.capture(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
    )
    print(f"[Rank {rank}] capture passed for {model_path}")
    del output, target, input_ids, attention_mask, loss_mask
    cleanup_distributed()


def test_dense(rank, world_size, port, tp_size):
    _run_capture(rank, world_size, port, tp_size, "unsloth/Llama-3.2-1B")


def test_moe(rank, world_size, port, tp_size):
    _run_capture(
        rank,
        world_size,
        port,
        tp_size,
        "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        load_format="dummy",
    )


@torch.no_grad()
def test_vlm(rank, world_size, port, tp_size):
    """Exercise the canonical multimodal capture contract across target TP."""
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    init_distributed(tp_size=tp_size)
    set_seed(42)

    from PIL import Image
    from transformers import AutoProcessor

    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_path)
    image = Image.new("RGB", (56, 56), color=(127, 63, 31))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the image."},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()
    loss_mask = attention_mask.clone()
    pixel_values = inputs["pixel_values"].cuda()
    image_grid_thw = inputs["image_grid_thw"].cuda()

    target = SGLangTargetEngine.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device="cuda",
        attention_backend="fa3",
        load_format="dummy",
        mem_fraction_static=0.75,
    )
    target.set_capture_layers()
    output = target.capture(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        media_inputs=MediaInputs(
            pixel_values=pixel_values,
            image_grid_thw=(image_grid_thw,),
        ),
    )
    position_ids, _ = target.get_rope_index(
        input_ids=output.input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=output.attention_mask,
    )

    assert output.hidden_states.shape[:2] == input_ids.shape
    assert output.target.shape[:2] == input_ids.shape
    assert position_ids.shape[-1] == input_ids.shape[-1]
    print(f"[Rank {rank}] VLM capture passed for {model_path}")
    del output, target, inputs, pixel_values, image_grid_thw
    del input_ids, attention_mask, loss_mask, processor, image
    cleanup_distributed()


class TestTargetModelBackend(unittest.TestCase):
    def test_sglang_backend_with_dense(self):
        world_size = 2
        mp.spawn(
            test_dense,
            nprocs=world_size,
            args=(world_size, get_available_port(), 2),
        )

    def test_sglang_backend_with_moe(self):
        world_size = 2
        mp.spawn(
            test_moe,
            nprocs=world_size,
            args=(world_size, get_available_port(), 2),
        )

    def test_sglang_backend_with_vlm(self):
        world_size = 2
        mp.spawn(
            test_vlm,
            nprocs=world_size,
            args=(world_size, get_available_port(), 2),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
