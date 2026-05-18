#!/usr/bin/env python3
"""Launch a standalone target model inference server for remote SpecForge training.

Usage
-----
Eagle3:
  python -m torch.distributed.run --nproc_per_node=1 \\
      scripts/launch_target_server.py \\
      --model-path /path/to/target/model \\
      --mode eagle3 --port 8001 --tp-size 1

DFlash:
  python -m torch.distributed.run --nproc_per_node=1 \\
      scripts/launch_target_server.py \\
      --model-path /path/to/target/model \\
      --mode dflash --port 8002 --tp-size 1

The server loads the target model via SGLang and exposes HTTP endpoints
that training scripts can call to generate Eagle3 / DFlash training data.
"""

import argparse
import logging
import os
import signal
import sys

import torch
import torch.distributed as dist

# Add parent dir so 'specforge' is importable when running as a standalone script.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SpecForge target model inference server"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the target model (e.g. /workspace/data/model/Qwen3-30B-A3B-Instruct-2507-FP8)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["eagle3", "dflash"],
        help="Target model mode: eagle3 or dflash",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="HTTP server port (default: 8001)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="HTTP server bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallelism size (default: 1)",
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.85,
        help="Fraction of GPU memory for static allocation (default: 0.85)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading the model",
    )
    parser.add_argument(
        "--enable-torch-compile",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Enable torch.compile for target model (default: True, matches baseline)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--nccl-port",
        type=int,
        default=None,
        help="NCCL TCP rendezvous port for GPU-to-GPU data transfer (default: HTTP port + 100)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, args.log_level),
    )
    logger = logging.getLogger("launch_target_server")

    # ---- Initialize distributed (required by SGLang internals) ----
    # The SGLang target model wrapper calls specforge.distributed.get_tp_group()
    # which requires init_process_group. Always run via torchrun.
    from specforge.distributed import init_distributed
    init_distributed(timeout=30, tp_size=args.tp_size)
    logger.info("Distributed initialized (tp_size=%d, rank=%d)", args.tp_size, dist.get_rank())

    # ---- Load model via TargetModelServer ----
    from specforge.modeling.target.remote_target_server import (
        TargetModelServer,
        create_http_server,
    )

    logger.info(
        "Loading %s target model from %s (tp_size=%d, mem_fraction_static=%.2f)...",
        args.mode,
        args.model_path,
        args.tp_size,
        args.mem_fraction_static,
    )

    server_app = TargetModelServer(
        mode=args.mode,
        model_path=args.model_path,
        tp_size=args.tp_size,
        mem_fraction_static=args.mem_fraction_static,
        trust_remote_code=args.trust_remote_code,
        enable_torch_compile=args.enable_torch_compile,
        nccl_port=args.nccl_port if args.nccl_port else args.port + 100,
        host=args.host,
    )
    server_app.load_model()
    logger.info("Model loaded successfully.")

    rank = dist.get_rank()

    if rank == 0:
        # ---- Start HTTP server (rank 0 only) ----
        httpd = create_http_server(server_app, args.host, args.port)
        logger.info("Target model server listening on %s:%d (mode=%s)", args.host, args.port, args.mode)

        def shutdown(signum, frame):
            logger.info("Received signal %d, shutting down...", signum)
            httpd.shutdown()

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()
            # Signal worker ranks to exit
            if dist.get_world_size() > 1:
                from specforge.modeling.target.remote_target_server import _SENTINEL_EXIT

                dist.broadcast_object_list([_SENTINEL_EXIT], src=0)
        logger.info("Server stopped.")
    else:
        # ---- Worker ranks: participate in NCCL-synchronized forward passes ----
        from specforge.modeling.target.remote_target_server import _worker_loop

        logger.info("Worker rank %d entering sync loop", rank)
        _worker_loop(server_app)


if __name__ == "__main__":
    main()
