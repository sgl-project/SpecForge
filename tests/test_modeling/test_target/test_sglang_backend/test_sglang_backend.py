import os
import unittest

import torch
import torch.multiprocessing as mp
from accelerate.utils import set_seed

from specforge.distributed import init_distributed
from specforge.modeling.target.eagle3_target_model import SGLangEagle3TargetModel
from tests.utils import get_available_port


@torch.no_grad()
def test_dense(rank, world_size, port, tp_size):
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

    # test dense model
    sgl_target_model = SGLangEagle3TargetModel.from_pretrained(
        "unsloth/Llama-3.2-1B",
        torch_dtype=torch.float16,
        device="cuda",
        attention_backend="fa3",
        mem_fraction_static=0.4,
        enable_torch_compile=True,
        enable_nccl_nvls=True,
        enable_symm_mem=True,
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        enable_piecewise_cuda_graph=True,
        ep_size=1,
        context_length=256,
    )
    sgl_target_model.set_aux_hidden_states_layers()
    sgl_out = sgl_target_model.generate_eagle3_data(
        input_ids=input_ids, attention_mask=attention_mask, loss_mask=loss_mask
    )


@torch.no_grad()
def test_moe(rank, world_size, port, tp_size):
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

    # test moe model
    sgl_target_model = SGLangEagle3TargetModel.from_pretrained(
        "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        torch_dtype=torch.float16,
        device="cuda",
        attention_backend="fa3",
        mem_fraction_static=0.4,
        enable_torch_compile=True,
        enable_nccl_nvls=True,
        enable_symm_mem=True,
        enable_dp_attention=True,
        enable_dp_lm_head=True,
        enable_piecewise_cuda_graph=True,
        ep_size=2,
        context_length=256,
    )
    sgl_target_model.set_aux_hidden_states_layers()
    sgl_out = sgl_target_model.generate_eagle3_data(
        input_ids=input_ids, attention_mask=attention_mask, loss_mask=loss_mask
    )


def test_vlm(rank, world_size, port, tp_size):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    init_distributed(tp_size=tp_size)
    set_seed(42)

    model_path = "Qwen/Qwen2.5-VL-32B-Instruct"

    # Use Qwen2.5-VL processor to prepare inputs
    from qwen_vl_utils import process_vision_info
    from transformers import Qwen2_5_VLProcessor

    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)

    # Create test messages with images (batch_size=2)
    # Sample 1: single image
    messages_1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "./images/demo.jpeg"},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Sample 2: single image (can use same or different image)
    messages_2 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "./images/demo.jpeg"},
                {"type": "text", "text": "What do you see in this picture?"},
            ],
        }
    ]

    # Process each sample separately to get correct format
    batch_input_ids = []
    batch_attention_mask = []
    batch_pixel_values = []
    batch_image_grid_thw = []

    for messages in [messages_1, messages_2]:
        # Apply chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Process vision info to get actual image data
        image_inputs, video_inputs = process_vision_info(messages)

        # Process with processor
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        batch_input_ids.append(inputs["input_ids"])
        batch_attention_mask.append(inputs["attention_mask"])
        batch_pixel_values.append(inputs["pixel_values"])
        batch_image_grid_thw.append(inputs["image_grid_thw"])

    # Debug: print shapes
    if rank == 0:
        print(f"[Debug] batch_input_ids shapes: {[x.shape for x in batch_input_ids]}")
        print(
            f"[Debug] batch_pixel_values shapes: {[x.shape for x in batch_pixel_values]}"
        )
        print(
            f"[Debug] batch_image_grid_thw shapes: {[x.shape for x in batch_image_grid_thw]}"
        )
        print(f"[Debug] batch_image_grid_thw values: {batch_image_grid_thw}")
        # Count image tokens in input_ids
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        for i, ids in enumerate(batch_input_ids):
            num_img_tokens = (ids == image_token_id).sum().item()
            print(f"[Debug] Sample {i}: {num_img_tokens} image tokens in input_ids")

    # Pad input_ids and attention_mask to same length
    max_len = max(ids.shape[1] for ids in batch_input_ids)
    padded_input_ids = []
    padded_attention_mask = []
    padded_loss_mask = []

    for input_ids, attention_mask in zip(batch_input_ids, batch_attention_mask):
        pad_len = max_len - input_ids.shape[1]
        if pad_len > 0:
            input_ids = torch.nn.functional.pad(
                input_ids, (0, pad_len), value=processor.tokenizer.pad_token_id
            )
            attention_mask = torch.nn.functional.pad(
                attention_mask, (0, pad_len), value=0
            )
        padded_input_ids.append(input_ids)
        padded_attention_mask.append(attention_mask)
        padded_loss_mask.append(
            attention_mask.clone()
        )  # loss_mask same as attention_mask

    # Stack into batches
    input_ids = torch.cat(padded_input_ids, dim=0).cuda()
    attention_mask = torch.cat(padded_attention_mask, dim=0).cuda()
    loss_mask = torch.cat(padded_loss_mask, dim=0).cuda()

    # pixel_values and image_grid_thw remain as lists (one per sample)
    pixel_values = torch.cat(batch_pixel_values, dim=0).cuda()
    image_grid_thw = [thw.cuda() for thw in batch_image_grid_thw]

    sgl_target_model = SGLangEagle3TargetModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device="cuda",
        attention_backend="fa3",
        mem_fraction_static=0.75,
        enable_torch_compile=True,
        enable_nccl_nvls=False,
        enable_symm_mem=False,  # Disable to avoid nccl_allocator compilation issues
        enable_dp_attention=True,
        enable_dp_lm_head=True,
        enable_piecewise_cuda_graph=True,
        context_length=4096,
    )
    sgl_target_model.set_aux_hidden_states_layers()
    sgl_out = sgl_target_model.generate_eagle3_data(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        is_vlm=True,
    )

    if rank == 0:
        # Verify output shapes
        print(f"[Rank {rank}] hidden_states shape: {sgl_out.hidden_states.shape}")
        print(f"[Rank {rank}] target shape: {sgl_out.target.shape}")
        print(f"[Rank {rank}] input_ids shape: {sgl_out.input_ids.shape}")


def test_vlm_multi_batch(rank, world_size, port, tp_size):
    """Test VLM with larger batch size (4 samples) and varying image counts."""
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    init_distributed(tp_size=tp_size)
    set_seed(42)

    model_path = "/home/qspace/Qwen2.5-VL-32B-Instruct"

    from qwen_vl_utils import process_vision_info
    from transformers import Qwen2_5_VLProcessor

    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)

    image_path = os.path.join(os.path.dirname(__file__), "images", "demo.jpeg")

    # Create test messages with different configurations (batch_size=4)
    # Sample 1: single image
    messages_1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ]

    # Sample 2: single image with different prompt
    messages_2 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "What objects can you see in this picture?"},
            ],
        }
    ]

    # Sample 3: single image with longer prompt
    messages_3 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {
                    "type": "text",
                    "text": "Please analyze this image and describe the main subject, background, colors, and any notable details you observe.",
                },
            ],
        }
    ]

    # Sample 4: single image with short prompt
    messages_4 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "What is this?"},
            ],
        }
    ]

    all_messages = [messages_1, messages_2, messages_3, messages_4]
    batch_size = len(all_messages)

    # Process each sample separately to get correct format
    batch_input_ids = []
    batch_attention_mask = []
    batch_pixel_values = []
    batch_image_grid_thw = []

    for messages in all_messages:
        # Apply chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Process vision info to get actual image data
        image_inputs, video_inputs = process_vision_info(messages)

        # Process with processor
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        batch_input_ids.append(inputs["input_ids"])
        batch_attention_mask.append(inputs["attention_mask"])
        batch_pixel_values.append(inputs["pixel_values"])
        batch_image_grid_thw.append(inputs["image_grid_thw"])

    # Debug: print shapes
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"[test_vlm_multi_batch] batch_size = {batch_size}")
        print(f"[Debug] batch_input_ids shapes: {[x.shape for x in batch_input_ids]}")
        print(
            f"[Debug] batch_pixel_values shapes: {[x.shape for x in batch_pixel_values]}"
        )
        print(
            f"[Debug] batch_image_grid_thw shapes: {[x.shape for x in batch_image_grid_thw]}"
        )
        print(f"[Debug] batch_image_grid_thw values: {batch_image_grid_thw}")
        # Count image tokens in input_ids
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        for i, ids in enumerate(batch_input_ids):
            num_img_tokens = (ids == image_token_id).sum().item()
            print(
                f"[Debug] Sample {i}: {num_img_tokens} image tokens, seq_len={ids.shape[1]}"
            )
        print(f"{'='*60}\n")

    # Pad input_ids and attention_mask to same length
    max_len = max(ids.shape[1] for ids in batch_input_ids)
    padded_input_ids = []
    padded_attention_mask = []
    padded_loss_mask = []

    for input_ids, attention_mask in zip(batch_input_ids, batch_attention_mask):
        pad_len = max_len - input_ids.shape[1]
        if pad_len > 0:
            input_ids = torch.nn.functional.pad(
                input_ids, (0, pad_len), value=processor.tokenizer.pad_token_id
            )
            attention_mask = torch.nn.functional.pad(
                attention_mask, (0, pad_len), value=0
            )
        padded_input_ids.append(input_ids)
        padded_attention_mask.append(attention_mask)
        padded_loss_mask.append(
            attention_mask.clone()
        )  # loss_mask same as attention_mask

    # Stack into batches
    input_ids = torch.cat(padded_input_ids, dim=0).cuda()
    attention_mask = torch.cat(padded_attention_mask, dim=0).cuda()
    loss_mask = torch.cat(padded_loss_mask, dim=0).cuda()

    # pixel_values and image_grid_thw remain as lists (one per sample)
    pixel_values = torch.cat(batch_pixel_values, dim=0).cuda()
    image_grid_thw = [thw.cuda() for thw in batch_image_grid_thw]

    if rank == 0:
        print(f"[Debug] Final input_ids shape: {input_ids.shape}")
        print(f"[Debug] Final attention_mask shape: {attention_mask.shape}")
        print(f"[Debug] Final pixel_values list length: {len(pixel_values)}")
        print(f"[Debug] Final image_grid_thw list length: {len(image_grid_thw)}")

    sgl_target_model = SGLangEagle3TargetModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device="cuda",
        attention_backend="fa3",
        mem_fraction_static=0.4,
        enable_torch_compile=True,
        enable_nccl_nvls=False,
        enable_symm_mem=False,
        enable_dp_attention=True,
        enable_dp_lm_head=True,
        enable_piecewise_cuda_graph=True,
        context_length=4096,
    )
    sgl_target_model.set_aux_hidden_states_layers()
    sgl_out = sgl_target_model.generate_eagle3_data(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        is_vlm=True,
    )

    if rank == 0:
        # Verify output shapes
        print(f"\n{'='*60}")
        print(f"[test_vlm_multi_batch] Results:")
        print(f"[Rank {rank}] hidden_states shape: {sgl_out.hidden_states.shape}")
        print(f"[Rank {rank}] target shape: {sgl_out.target.shape}")
        print(f"[Rank {rank}] input_ids shape: {sgl_out.input_ids.shape}")

        # Verify batch dimension matches
        assert (
            sgl_out.input_ids.shape[0] == batch_size
        ), f"Expected batch_size={batch_size}, got {sgl_out.input_ids.shape[0]}"
        print(f"[Rank {rank}] Batch size verification: PASSED")
        print(f"{'='*60}\n")


class TestTargetModelBackend(unittest.TestCase):

    def test_sglang_backend_with_dense(self):
        world_size = 2
        port = get_available_port()
        mp.spawn(test_dense, nprocs=world_size, args=(world_size, port, 2))

    def test_sglang_backend_with_moe(self):
        world_size = 2
        port = get_available_port()
        mp.spawn(test_moe, nprocs=world_size, args=(world_size, port, 2))

    def test_sglang_backend_with_vlm(self):
        world_size = 4
        port = get_available_port()
        mp.spawn(test_vlm, nprocs=world_size, args=(world_size, port, 4))

    def test_sglang_backend_with_vlm_multi_batch(self):
        world_size = 8
        port = get_available_port()
        mp.spawn(test_vlm_multi_batch, nprocs=world_size, args=(world_size, port, 8))


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestTargetModelBackend))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
