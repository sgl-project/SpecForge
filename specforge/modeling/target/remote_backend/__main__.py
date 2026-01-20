#!/usr/bin/env python3
"""
Entry point for running inference worker as a module.

Usage:
    python -m specforge.modeling.target.remote_backend --model-path <PATH> [OPTIONS]
"""

import argparse
import logging
import os

from specforge.modeling.utils import parse_mooncake_device_name

from ..sglang_backend import SGLangBackendArgs
from .inference_worker import InferenceWorkerConfig, run_worker
from .mooncake_client import MooncakeConfig
from .task_queue import QueueConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference worker for distributed Eagle3 training"
    )

    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--tp-size", type=int, default=1, help="Tensor parallelism size"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype")
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code"
    )

    SGLangBackendArgs.add_args(parser)

    parser.add_argument(
        "--task-queue-addr",
        type=str,
        default="tcp://localhost:5555",
        help="Task queue address",
    )
    parser.add_argument(
        "--notify-addr",
        type=str,
        default="tcp://localhost:5556",
        help="Notification address",
    )

    parser.add_argument(
        "--mooncake-master-addr",
        type=str,
        default="localhost:50051",
        help="Mooncake master address",
    )
    parser.add_argument(
        "--mooncake-metadata-port",
        type=int,
        default=8080,
        help="Mooncake metadata port",
    )
    parser.add_argument(
        "--mooncake-protocol", type=str, default="tcp", help="Mooncake protocol"
    )
    parser.add_argument(
        "--mooncake-device-name",
        type=str,
        default="",
        help=(
            "RDMA device name. Supports: "
            "1) Simple format: 'mlx5_0,mlx5_1' (same for all GPUs), "
            "2) JSON file: '/path/to/mapping.json' (per-GPU mapping)"
        ),
    )
    parser.add_argument(
        "--mooncake-global-segment-size",
        type=str,
        default="4GB",
        help="Global segment size",
    )
    parser.add_argument(
        "--mooncake-local-buffer-size",
        type=str,
        default="512MB",
        help="Local buffer size",
    )

    parser.add_argument(
        "--use-zero-copy",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Use zero-copy transfers",
    )
    parser.add_argument("--worker-id", type=int, default=0, help="Worker ID")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    master_host = args.mooncake_master_addr.split(":")[0]
    metadata_server = f"http://{master_host}:{args.mooncake_metadata_port}/metadata"

    # Parse device name - supports simple string or JSON file with per-GPU mapping
    try:
        device_name = parse_mooncake_device_name(
            args.mooncake_device_name, args.worker_id
        )
    except Exception as e:
        logging.error(f"Failed to parse mooncake device name: {e}")
        raise

    mooncake_config = MooncakeConfig(
        local_hostname=os.getenv("HOSTNAME", "localhost"),
        metadata_server=metadata_server,
        master_server_address=args.mooncake_master_addr,
        global_segment_size=MooncakeConfig.parse_size(
            args.mooncake_global_segment_size
        ),
        local_buffer_size=MooncakeConfig.parse_size(args.mooncake_local_buffer_size),
        protocol=args.mooncake_protocol,
        device_name=device_name,
    )

    queue_config = QueueConfig(
        task_queue_addr=args.task_queue_addr,
        notify_addr=args.notify_addr,
    )

    sglang_backend_args = SGLangBackendArgs.from_args(args)

    config = InferenceWorkerConfig(
        model_path=args.model_path,
        task_queue_addr=args.task_queue_addr,
        notify_addr=args.notify_addr,
        tp_size=args.tp_size,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        worker_id=args.worker_id,
        mooncake_config=mooncake_config,
        queue_config=queue_config,
        sglang_backend_args=sglang_backend_args,
    )

    logging.info(f"Starting inference worker {args.worker_id}")
    logging.info(f"Model: {args.model_path}")
    logging.info(f"Task queue: {args.task_queue_addr}")

    run_worker(config)


if __name__ == "__main__":
    main()
