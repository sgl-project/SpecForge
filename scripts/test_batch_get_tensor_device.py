#!/usr/bin/env python3
"""
Test script to verify get_eagle3_tensors_into retrieves tensors to GPU.

Supports both RDMA (GPUDirect) and TCP (via host buffer fallback).

Usage:
    # Start mooncake master first, then run:
    python scripts/test_batch_get_tensor_device.py
"""

import time
import torch

from specforge.modeling.target.remote_backend.mooncake_client import (
    EagleMooncakeStore,
    MooncakeConfig,
)


def test_batch_get_tensor_device():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    config = MooncakeConfig.from_env()
    store = EagleMooncakeStore(config)
    store.setup()

    cuda_device = torch.device("cuda:0")

    try:
        batch_size, seq_len, hidden_dim, vocab_size = 2, 128, 256, 1000

        hidden_states = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)
        target = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.bfloat16)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.int64)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int64)
        last_hidden_states = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)

        test_key = f"test-batch-get-device-{int(time.time())}"

        print("Storing Eagle3 tensors...")
        print(f"  hidden_states: {hidden_states.shape}, {hidden_states.dtype}")
        print(f"  target: {target.shape}, {target.dtype}")
        shapes = store.put_eagle3_tensors(
            key=test_key,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            last_hidden_states=last_hidden_states,
            target=target,
        )
        print(f"Stored tensors with shapes: {shapes}")

        dtypes = {
            "hidden_states": torch.bfloat16,
            "target": torch.bfloat16,
        }
        shapes["target"] = tuple(target.shape)
        shapes["last_hidden_states"] = tuple(last_hidden_states.shape)

        print("\n--- Test 1: Get tensors into GPU ---")
        print("(Uses GPUDirect RDMA if available, falls back to TCP/host buffer)")
        output = store.get_eagle3_tensors_into(
            key=test_key, shapes=shapes, dtypes=dtypes, device=cuda_device
        )

        assert output.hidden_states.device == cuda_device
        assert output.target.device == cuda_device
        assert output.loss_mask.device == cuda_device
        assert output.input_ids.device == cuda_device
        assert output.attention_mask.device == cuda_device
        assert output.last_hidden_states.device == cuda_device
        print("✓ All tensors correctly on CUDA")

        print("\n--- Test 2: Verify tensor shapes ---")
        assert output.hidden_states.shape == hidden_states.shape
        assert output.target.shape == target.shape
        assert output.loss_mask.shape == loss_mask.shape
        assert output.input_ids.shape == input_ids.shape
        assert output.attention_mask.shape == attention_mask.shape
        assert output.last_hidden_states.shape == last_hidden_states.shape
        print("✓ All tensor shapes match")

        print("\n--- Test 3: Verify tensor dtypes ---")
        assert output.hidden_states.dtype == torch.bfloat16
        assert output.target.dtype == torch.bfloat16
        assert output.loss_mask.dtype == torch.bool
        assert output.input_ids.dtype == torch.int64
        assert output.attention_mask.dtype == torch.int64
        print("✓ All tensor dtypes match")

        print("\n--- Test 4: Multiple retrievals ---")
        output2 = store.get_eagle3_tensors_into(
            key=test_key, shapes=shapes, dtypes=dtypes, device=cuda_device
        )
        assert output2.hidden_states.device == cuda_device
        print("✓ Multiple retrievals work")

        store.remove_eagle3_tensors(
            key=test_key, has_last_hidden_states=True, has_target=True
        )
        print("\n✓ Cleaned up test tensors")

        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)

    finally:
        store.close()


if __name__ == "__main__":
    test_batch_get_tensor_device()
