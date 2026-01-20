#!/usr/bin/env python
"""
Example script to send inference tasks to a remote backend worker.

Usage:
    1. Start the inference worker: ./examples/run_remote_inference.sh
    2. Run this script: python examples/send_remote_task.py
"""

import torch
import logging

from specforge.modeling.target.remote_backend import (
    RemoteEagle3TargetModel,
    RemoteBackendConfig,
    MooncakeConfig,
)

logging.basicConfig(level=logging.INFO)


def main():
    mooncake_config = MooncakeConfig(
        local_hostname="10.173.2.69",
        metadata_server="http://localhost:8090/metadata",
        master_server_address="127.0.0.1:50051",
        global_segment_size=512 * 1024 * 1024,
        local_buffer_size=128 * 1024 * 1024,
        protocol="rdma",
        device_name="mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_12,mlx5_13",
    )

    config = RemoteBackendConfig(
        task_queue_addr="tcp://127.0.0.1:5555",
        notify_addr="tcp://127.0.0.1:5556",
        task_timeout=300.0,
        mooncake_config=mooncake_config,
        target_model_path="Qwen/Qwen3-8B",
    )

    model = RemoteEagle3TargetModel(config=config)
    model.connect()

    batch_size = 1
    seq_len = 128
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), dtype=torch.long, device="cuda")
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device="cuda")
    loss_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device="cuda")

    print(f"Submitting task with input shape: {input_ids.shape}")

    output = model.generate_eagle3_data(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
    )

    print(f"Received output:")
    print(f"  - hidden_states: {[h.shape for h in output.hidden_states]}")
    print(f"  - target: {output.target.shape}")
    print(f"  - loss_mask: {output.loss_mask.shape}")
    print(f"  - input_ids: {output.input_ids.shape}")

    model.disconnect()
    print("Done!")


if __name__ == "__main__":
    main()
