#!/usr/bin/env python
"""
Compare local sglang backend inference with remote backend inference.

This script verifies that the remote inference results are the same as local
sglang backend results.

Usage:
    1. Start the remote inference worker: ./examples/run_remote_inference.sh
    2. Run this script on a separate GPU:
       CUDA_VISIBLE_DEVICES=1 python examples/compare_local_remote_inference.py
"""

import argparse
import logging
import torch

from specforge.modeling.target.sglang_backend import (
    init_sglang_distributed,
    destroy_sglang_distributed,
)
from specforge.modeling.target.eagle3_target_model import SGLangEagle3TargetModel
from specforge.modeling.target.remote_backend import (
    RemoteEagle3TargetModel,
    RemoteBackendConfig,
    MooncakeConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare local and remote inference")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model path for local inference",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for testing",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length for testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--task-queue-addr",
        type=str,
        default="tcp://127.0.0.1:5555",
        help="ZMQ task queue address",
    )
    parser.add_argument(
        "--notify-addr",
        type=str,
        default="tcp://127.0.0.1:5556",
        help="ZMQ notification address",
    )
    parser.add_argument(
        "--mooncake-local-hostname",
        type=str,
        default="10.173.2.69",
        help="Local hostname for Mooncake",
    )
    parser.add_argument(
        "--mooncake-metadata-server",
        type=str,
        default="http://localhost:8090/metadata",
        help="Mooncake metadata server URL",
    )
    parser.add_argument(
        "--mooncake-master-addr",
        type=str,
        default="127.0.0.1:50051",
        help="Mooncake master server address",
    )
    parser.add_argument(
        "--mooncake-protocol",
        type=str,
        default="rdma",
        help="Mooncake protocol (tcp or rdma)",
    )
    parser.add_argument(
        "--mooncake-device-name",
        type=str,
        default="mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_12,mlx5_13",
        help="RDMA device names",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for comparison",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for comparison",
    )
    return parser.parse_args()


def compare_tensors(name: str, local: torch.Tensor, remote: torch.Tensor, atol: float, rtol: float) -> bool:
    """Compare two tensors and report differences."""
    if local is None and remote is None:
        logger.info(f"  {name}: Both None (OK)")
        return True
    
    if local is None or remote is None:
        logger.error(f"  {name}: One is None, other is not!")
        return False
    
    local_shape = local.shape
    remote_shape = remote.shape
    
    if local_shape != remote_shape:
        local_squeezed = local.squeeze()
        remote_squeezed = remote.squeeze()
        if local_squeezed.shape == remote_squeezed.shape:
            logger.warning(f"  {name}: Shape differs but squeezed shapes match - local {local_shape} vs remote {remote_shape}")
            local = local_squeezed
            remote = remote_squeezed
        else:
            logger.error(f"  {name}: Shape mismatch - local {local_shape} vs remote {remote_shape}")
            return False
    
    if local.dtype != remote.dtype:
        logger.info(f"  {name}: dtype differs - local {local.dtype} vs remote {remote.dtype} (converting for comparison)")
    
    local_dev = local.to("cpu").float()
    remote_dev = remote.to("cpu").float()
    
    is_close = torch.allclose(local_dev, remote_dev, atol=atol, rtol=rtol)
    max_diff = (local_dev - remote_dev).abs().max().item()
    mean_diff = (local_dev - remote_dev).abs().mean().item()
    
    if is_close:
        logger.info(f"  {name}: MATCH (max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e})")
    else:
        logger.error(f"  {name}: MISMATCH (max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e})")
    
    return is_close


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    logger.info("=" * 60)
    logger.info("Initializing local sglang backend...")
    logger.info("=" * 60)
    
    init_sglang_distributed(tp_size=1)
    
    local_model = SGLangEagle3TargetModel.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    local_model.set_aux_hidden_states_layers([1, -1, -4])
    
    logger.info("Local model loaded successfully")
    
    logger.info("=" * 60)
    logger.info("Setting up remote backend connection...")
    logger.info("=" * 60)
    
    mooncake_config = MooncakeConfig(
        local_hostname=args.mooncake_local_hostname,
        metadata_server=args.mooncake_metadata_server,
        master_server_address=args.mooncake_master_addr,
        global_segment_size=512 * 1024 * 1024,
        local_buffer_size=128 * 1024 * 1024,
        protocol=args.mooncake_protocol,
        device_name=args.mooncake_device_name,
    )
    
    remote_config = RemoteBackendConfig(
        task_queue_addr=args.task_queue_addr,
        notify_addr=args.notify_addr,
        task_timeout=300.0,
        mooncake_config=mooncake_config,
        target_model_path=args.model_path,
    )
    
    remote_model = RemoteEagle3TargetModel(config=remote_config, aux_hidden_states_layers=[1, -1, -4])
    remote_model.connect()
    
    logger.info("Remote backend connected successfully")
    
    logger.info("=" * 60)
    logger.info("Generating test input...")
    logger.info("=" * 60)
    
    input_ids = torch.randint(
        0, 32000, (args.batch_size, args.seq_len), dtype=torch.long, device="cuda"
    )
    attention_mask = torch.ones(
        (args.batch_size, args.seq_len), dtype=torch.long, device="cuda"
    )
    loss_mask = torch.ones(
        (args.batch_size, args.seq_len), dtype=torch.bool, device="cuda"
    )
    
    logger.info(f"Input shape: {input_ids.shape}")
    logger.info(f"Input IDs (first 10): {input_ids[0, :10].tolist()}")
    
    logger.info("=" * 60)
    logger.info("Running local inference...")
    logger.info("=" * 60)
    
    with torch.no_grad():
        local_output = local_model.generate_eagle3_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )
    
    logger.info(f"Local output shapes:")
    logger.info(f"  hidden_states: {local_output.hidden_states.shape}")
    logger.info(f"  target: {local_output.target.shape}")
    logger.info(f"  loss_mask: {local_output.loss_mask.shape}")
    logger.info(f"  input_ids: {local_output.input_ids.shape}")
    
    logger.info("=" * 60)
    logger.info("Running remote inference...")
    logger.info("=" * 60)
    
    remote_output = remote_model.generate_eagle3_data(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
    )
    
    logger.info(f"Remote output shapes:")
    logger.info(f"  hidden_states: {remote_output.hidden_states.shape}")
    logger.info(f"  target: {remote_output.target.shape}")
    logger.info(f"  loss_mask: {remote_output.loss_mask.shape}")
    logger.info(f"  input_ids: {remote_output.input_ids.shape}")
    
    logger.info("=" * 60)
    logger.info("Comparing outputs...")
    logger.info("=" * 60)
    
    all_match = True
    
    all_match &= compare_tensors(
        "hidden_states", 
        local_output.hidden_states, 
        remote_output.hidden_states,
        args.atol, args.rtol
    )
    
    all_match &= compare_tensors(
        "target (logits)",
        local_output.target,
        remote_output.target,
        args.atol, args.rtol
    )
    
    all_match &= compare_tensors(
        "loss_mask",
        local_output.loss_mask,
        remote_output.loss_mask,
        args.atol, args.rtol
    )
    
    all_match &= compare_tensors(
        "input_ids",
        local_output.input_ids,
        remote_output.input_ids,
        args.atol, args.rtol
    )
    
    logger.info("=" * 60)
    if all_match:
        logger.info("SUCCESS: All outputs match between local and remote inference!")
    else:
        logger.error("FAILURE: Some outputs do not match!")
    logger.info("=" * 60)
    
    remote_model.disconnect()
    destroy_sglang_distributed()
    
    logger.info("Done!")
    return 0 if all_match else 1


if __name__ == "__main__":
    exit(main())
