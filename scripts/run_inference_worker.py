#!/usr/bin/env python3
"""
Run an inference worker for distributed Eagle3 training.

This script starts an inference worker that:
1. Pulls inference tasks from a ZeroMQ queue
2. Runs inference using SGLang's Offline Engine API
3. Extracts hidden states and stores them in Mooncake Store
4. Notifies training nodes when results are ready

Usage:
    python run_inference_worker.py --model-path <MODEL_PATH> [OPTIONS]

Example:
    python run_inference_worker.py \
        --model-path Qwen/Qwen3-8B \
        --task-queue-addr tcp://master:5555 \
        --notify-addr tcp://master:5556 \
        --mooncake-master-addr master:50051

For multi-GPU inference with tensor parallelism:
    python run_inference_worker.py \
        --model-path Qwen/Qwen3-70B \
        --tp-size 4 \
        --task-queue-addr tcp://master:5555
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specforge.modeling.target.remote_backend import (
    InferenceWorkerConfig,
    MooncakeConfig,
    QueueConfig,
    run_worker,
)
from specforge.modeling.target.sglang_backend import SGLangBackendArgs
from specforge.utils import parse_mooncake_device_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference worker for distributed Eagle3 training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model (HuggingFace model ID or local path)",
    )
    model_group.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallelism size",
    )
    model_group.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model data type",
    )
    model_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading model",
    )

    SGLangBackendArgs.add_args(parser)

    queue_group = parser.add_argument_group("Task Queue")
    queue_group.add_argument(
        "--task-queue-addr",
        type=str,
        default="tcp://localhost:5555",
        help="ZeroMQ address for task queue",
    )
    queue_group.add_argument(
        "--notify-addr",
        type=str,
        default="tcp://localhost:5556",
        help="ZeroMQ address for notifications",
    )

    mooncake_group = parser.add_argument_group("Mooncake Store")
    mooncake_group.add_argument(
        "--mooncake-local-hostname",
        type=str,
        default=None,
        help="Local hostname for Mooncake (auto-detected if not set)",
    )
    mooncake_group.add_argument(
        "--mooncake-master-addr",
        type=str,
        default="localhost:50051",
        help="Mooncake Master Service address (host:port)",
    )
    mooncake_group.add_argument(
        "--mooncake-metadata-server",
        type=str,
        default=None,
        help="Mooncake metadata server URL. If not set, uses the Master's built-in HTTP metadata server",
    )
    mooncake_group.add_argument(
        "--mooncake-metadata-port",
        type=int,
        default=8080,
        help="Port for Master's built-in HTTP metadata server (used when --mooncake-metadata-server is not set)",
    )
    mooncake_group.add_argument(
        "--mooncake-global-segment-size",
        type=str,
        default="4GB",
        help="Global segment size for Mooncake Store",
    )
    mooncake_group.add_argument(
        "--mooncake-local-buffer-size",
        type=str,
        default="512MB",
        help="Local buffer size for Mooncake Store",
    )
    mooncake_group.add_argument(
        "--mooncake-protocol",
        type=str,
        default="tcp",
        choices=["tcp", "rdma"],
        help="Protocol for Mooncake Store transfers",
    )
    mooncake_group.add_argument(
        "--mooncake-device-name",
        type=str,
        default="",
        help=(
            "RDMA device name. Supports: "
            "1) Simple format: 'mlx5_0,mlx5_1' (same for all GPUs), "
            "2) JSON file: '/path/to/mapping.json' (per-GPU mapping)"
        ),
    )

    worker_group = parser.add_argument_group("Worker")
    worker_group.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="Worker ID (for logging)",
    )
    worker_group.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Total number of workers (for logging)",
    )

    other_group = parser.add_argument_group("Other")
    other_group.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file (overrides command-line args)",
    )
    other_group.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def get_local_hostname():
    """Auto-detect local hostname."""
    import socket

    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        return ip
    except Exception:
        return "localhost"


def parse_size(size_str: str) -> int:
    """Parse size string like '4GB' to bytes."""
    size_str = size_str.upper().strip()
    # Ordered by suffix length (longest first) to avoid "4GB" matching "B" before "GB"
    multipliers = [
        ("TB", 1024 * 1024 * 1024 * 1024),
        ("GB", 1024 * 1024 * 1024),
        ("MB", 1024 * 1024),
        ("KB", 1024),
        ("T", 1024 * 1024 * 1024 * 1024),
        ("G", 1024 * 1024 * 1024),
        ("M", 1024 * 1024),
        ("K", 1024),
        ("B", 1),
    ]
    for suffix, multiplier in multipliers:
        if size_str.endswith(suffix):
            return int(float(size_str[: -len(suffix)]) * multiplier)
    return int(size_str)


def build_config_from_args(args) -> InferenceWorkerConfig:
    """Build InferenceWorkerConfig from command-line arguments."""
    local_hostname = args.mooncake_local_hostname or get_local_hostname()

    metadata_server = args.mooncake_metadata_server
    if metadata_server is None:
        master_host = args.mooncake_master_addr.split(":")[0]
        metadata_server = f"http://{master_host}:{args.mooncake_metadata_port}/metadata"

    # Parse device name - supports simple string or JSON file with per-GPU mapping
    device_name = parse_mooncake_device_name(args.mooncake_device_name, args.worker_id)

    mooncake_config = MooncakeConfig(
        local_hostname=local_hostname,
        metadata_server=metadata_server,
        master_server_address=args.mooncake_master_addr,
        global_segment_size=parse_size(args.mooncake_global_segment_size),
        local_buffer_size=parse_size(args.mooncake_local_buffer_size),
        protocol=args.mooncake_protocol,
        device_name=device_name,
    )

    queue_config = QueueConfig(
        task_queue_addr=args.task_queue_addr,
        notify_addr=args.notify_addr,
    )

    sglang_backend_args = SGLangBackendArgs.from_args(args)

    return InferenceWorkerConfig(
        model_path=args.model_path,
        task_queue_addr=args.task_queue_addr,
        notify_addr=args.notify_addr,
        tp_size=args.tp_size,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        num_workers=args.num_workers,
        worker_id=args.worker_id,
        mooncake_config=mooncake_config,
        queue_config=queue_config,
        sglang_backend_args=sglang_backend_args,
    )


def build_config_from_json(config_path: str) -> InferenceWorkerConfig:
    """Build InferenceWorkerConfig from JSON file."""
    with open(config_path) as f:
        config_dict = json.load(f)

    mooncake_dict = config_dict.get("mooncake", {})
    mooncake_config = MooncakeConfig(
        local_hostname=mooncake_dict.get("local_hostname", get_local_hostname()),
        metadata_server=mooncake_dict.get(
            "metadata_server", "http://localhost:8090/metadata"
        ),
        master_server_address=mooncake_dict.get(
            "master_server_address", "localhost:50051"
        ),
        global_segment_size=parse_size(mooncake_dict.get("global_segment_size", "4GB")),
        local_buffer_size=parse_size(mooncake_dict.get("local_buffer_size", "512MB")),
        protocol=mooncake_dict.get("protocol", "tcp"),
        device_name=mooncake_dict.get("device_name", ""),
    )

    queue_config = QueueConfig(
        task_queue_addr=config_dict.get("task_queue_addr", "tcp://localhost:5555"),
        notify_addr=config_dict.get("notify_addr", "tcp://localhost:5556"),
    )

    sglang_dict = config_dict.get("sglang_backend_args", {})
    sglang_backend_args = SGLangBackendArgs(
        sglang_attention_backend=sglang_dict.get("attention_backend", "flashinfer"),
        sglang_mem_fraction_static=sglang_dict.get("mem_fraction_static", 0.4),
        sglang_context_length=sglang_dict.get("context_length"),
        sglang_enable_nccl_nvls=sglang_dict.get("enable_nccl_nvls", False),
        sglang_enable_symm_mem=sglang_dict.get("enable_symm_mem", False),
        sglang_enable_torch_compile=sglang_dict.get("enable_torch_compile", True),
        sglang_disable_cuda_graph=sglang_dict.get("disable_cuda_graph", True),
        sglang_enable_dp_attention=sglang_dict.get("enable_dp_attention", False),
        sglang_enable_dp_lm_head=sglang_dict.get("enable_dp_lm_head", False),
        sglang_enable_piecewise_cuda_graph=sglang_dict.get(
            "enable_piecewise_cuda_graph", False
        ),
        sglang_piecewise_cuda_graph_max_tokens=sglang_dict.get(
            "piecewise_cuda_graph_max_tokens", 4096
        ),
        sglang_piecewise_cuda_graph_tokens=sglang_dict.get(
            "piecewise_cuda_graph_tokens"
        ),
        sglang_ep_size=sglang_dict.get("ep_size", 1),
        sglang_max_running_requests=sglang_dict.get("max_running_requests"),
        sglang_max_total_tokens=sglang_dict.get("max_total_tokens"),
    )

    return InferenceWorkerConfig(
        model_path=config_dict["model_path"],
        task_queue_addr=config_dict.get("task_queue_addr", "tcp://localhost:5555"),
        notify_addr=config_dict.get("notify_addr", "tcp://localhost:5556"),
        tp_size=config_dict.get("tp_size", 1),
        dtype=config_dict.get("dtype", "bfloat16"),
        trust_remote_code=config_dict.get("trust_remote_code", True),
        num_workers=config_dict.get("num_workers", 1),
        worker_id=config_dict.get("worker_id", 0),
        mooncake_config=mooncake_config,
        queue_config=queue_config,
        sglang_backend_args=sglang_backend_args,
    )


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting inference worker...")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Task queue: {args.task_queue_addr}")
    logger.info(f"Notification: {args.notify_addr}")

    if args.config:
        logger.info(f"Loading config from: {args.config}")
        if args.config.startswith("{"):
            import io

            config_dict = json.load(io.StringIO(args.config))
            with open("/tmp/worker_config.json", "w") as f:
                json.dump(config_dict, f)
            config = build_config_from_json("/tmp/worker_config.json")
        else:
            config = build_config_from_json(args.config)
    else:
        config = build_config_from_args(args)

    logger.info(f"Worker ID: {config.worker_id}/{config.num_workers}")
    logger.info(f"TP size: {config.tp_size}")
    logger.info(f"Mooncake master: {config.mooncake_config.master_server_address}")

    run_worker(config)


if __name__ == "__main__":
    main()
