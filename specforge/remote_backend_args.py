import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class RemoteBackendArgsMixin:
    """
    Arguments for remote inference backend using ZeroMQ and Mooncake Store.
    
    Note: The metadata server defaults to the Mooncake Master's built-in HTTP
    metadata server. If you only specify --mooncake-master-addr, the metadata
    server URL will be automatically derived (same host, port 8080).
    """

    task_queue_addr: str = "tcp://localhost:5555"
    notify_addr: str = "tcp://localhost:5556"
    task_timeout: float = 300.0
    prefetch_depth: int = 4
    use_zero_copy: bool = True
    zero_copy_buffer_size: str = "2GB"
    zero_copy_pool_size: int = 4
    dp_rank: int = 0
    dp_size: int = 1

    mooncake_local_hostname: str = "localhost"
    mooncake_metadata_server: str = None
    mooncake_master_addr: str = "localhost:50051"
    mooncake_metadata_port: int = 8080
    mooncake_global_segment_size: str = "4GB"
    mooncake_local_buffer_size: str = "512MB"
    mooncake_protocol: str = "tcp"
    mooncake_device_name: str = ""

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--task-queue-addr",
            type=str,
            default="tcp://localhost:5555",
            help="ZeroMQ address for task queue (PUSH/PULL)",
        )
        parser.add_argument(
            "--notify-addr",
            type=str,
            default="tcp://localhost:5556",
            help="ZeroMQ address for notification queue (PUB/SUB)",
        )
        parser.add_argument(
            "--task-timeout",
            type=float,
            default=300.0,
            help="Timeout in seconds for waiting for task completion",
        )
        parser.add_argument(
            "--prefetch-depth",
            type=int,
            default=4,
            help="Number of batches to prefetch to keep the inference pipeline saturated",
        )
        parser.add_argument(
            "--use-zero-copy",
            type=lambda x: x.lower() in ("true", "1", "yes"),
            default=True,
            help="Enable zero-copy GPU transfers for RDMA (default: True)",
        )
        parser.add_argument(
            "--zero-copy-buffer-size",
            type=str,
            default="2GB",
            help="Size of each GPU buffer for zero-copy transfers (e.g., '2GB')",
        )
        parser.add_argument(
            "--zero-copy-pool-size",
            type=int,
            default=4,
            help="Number of GPU buffers in the zero-copy pool",
        )
        parser.add_argument(
            "--dp-rank",
            type=int,
            default=None,
            help="Data parallel rank (auto-detected from RANK env var if not set)",
        )
        parser.add_argument(
            "--dp-size",
            type=int,
            default=None,
            help="Data parallel world size (auto-detected from WORLD_SIZE env var if not set)",
        )
        parser.add_argument(
            "--mooncake-local-hostname",
            type=str,
            default="localhost",
            help="Local hostname for Mooncake Store client",
        )
        parser.add_argument(
            "--mooncake-master-addr",
            type=str,
            default="localhost:50051",
            help="Mooncake Master Service address (host:port)",
        )
        parser.add_argument(
            "--mooncake-metadata-server",
            type=str,
            default=None,
            help="Mooncake metadata server URL. If not set, uses the Master's built-in HTTP metadata server (http://<master_host>:8080/metadata)",
        )
        parser.add_argument(
            "--mooncake-metadata-port",
            type=int,
            default=8080,
            help="Port for Master's built-in HTTP metadata server (used when --mooncake-metadata-server is not set)",
        )
        parser.add_argument(
            "--mooncake-global-segment-size",
            type=str,
            default="4GB",
            help="Global segment size for Mooncake Store (e.g., '4GB', '512MB')",
        )
        parser.add_argument(
            "--mooncake-local-buffer-size",
            type=str,
            default="512MB",
            help="Local buffer size for Mooncake Store (e.g., '512MB', '1GB')",
        )
        parser.add_argument(
            "--mooncake-protocol",
            type=str,
            default="tcp",
            choices=["tcp", "rdma"],
            help="Protocol for Mooncake Store transfers",
        )
        parser.add_argument(
            "--mooncake-device-name",
            type=str,
            default="",
            help="Device name for Mooncake Store (RDMA device, e.g., 'mlx5_0')",
        )

    @staticmethod
    def _get_dp_rank() -> int:
        """Auto-detect DP rank from environment variables."""
        for env_var in ["RANK", "LOCAL_RANK", "SLURM_PROCID"]:
            if env_var in os.environ:
                return int(os.environ[env_var])
        return 0

    @staticmethod
    def _get_dp_size() -> int:
        """Auto-detect DP world size from environment variables."""
        for env_var in ["WORLD_SIZE", "SLURM_NTASKS"]:
            if env_var in os.environ:
                return int(os.environ[env_var])
        return 1

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RemoteBackendArgsMixin":
        master_addr = getattr(args, "mooncake_master_addr", "localhost:50051")
        metadata_port = getattr(args, "mooncake_metadata_port", 8080)
        metadata_server = getattr(args, "mooncake_metadata_server", None)

        dp_rank = getattr(args, "dp_rank", None)
        if dp_rank is None:
            dp_rank = cls._get_dp_rank()

        dp_size = getattr(args, "dp_size", None)
        if dp_size is None:
            dp_size = cls._get_dp_size()

        return cls(
            task_queue_addr=getattr(args, "task_queue_addr", "tcp://localhost:5555"),
            notify_addr=getattr(args, "notify_addr", "tcp://localhost:5556"),
            task_timeout=getattr(args, "task_timeout", 300.0),
            prefetch_depth=getattr(args, "prefetch_depth", 4),
            use_zero_copy=getattr(args, "use_zero_copy", True),
            zero_copy_buffer_size=getattr(args, "zero_copy_buffer_size", "2GB"),
            zero_copy_pool_size=getattr(args, "zero_copy_pool_size", 4),
            dp_rank=dp_rank,
            dp_size=dp_size,
            mooncake_local_hostname=getattr(
                args, "mooncake_local_hostname", "localhost"
            ),
            mooncake_master_addr=master_addr,
            mooncake_metadata_server=metadata_server,
            mooncake_metadata_port=metadata_port,
            mooncake_global_segment_size=getattr(
                args, "mooncake_global_segment_size", "4GB"
            ),
            mooncake_local_buffer_size=getattr(
                args, "mooncake_local_buffer_size", "512MB"
            ),
            mooncake_protocol=getattr(args, "mooncake_protocol", "tcp"),
            mooncake_device_name=getattr(args, "mooncake_device_name", ""),
        )

    def to_kwargs(self) -> Dict[str, Any]:
        from specforge.modeling.target.remote_backend import (
            MooncakeConfig,
            RemoteBackendConfig,
        )

        metadata_server = self.mooncake_metadata_server
        if metadata_server is None:
            master_host = self.mooncake_master_addr.split(":")[0]
            metadata_server = f"http://{master_host}:{self.mooncake_metadata_port}/metadata"

        mooncake_config = MooncakeConfig(
            local_hostname=self.mooncake_local_hostname,
            metadata_server=metadata_server,
            master_server_address=self.mooncake_master_addr,
            global_segment_size=MooncakeConfig.parse_size(
                self.mooncake_global_segment_size
            ),
            local_buffer_size=MooncakeConfig.parse_size(self.mooncake_local_buffer_size),
            protocol=self.mooncake_protocol,
            device_name=self.mooncake_device_name,
        )

        return dict(
            remote_config=RemoteBackendConfig(
                task_queue_addr=self.task_queue_addr,
                notify_addr=self.notify_addr,
                task_timeout=self.task_timeout,
                use_zero_copy=self.use_zero_copy,
                zero_copy_buffer_size=MooncakeConfig.parse_size(self.zero_copy_buffer_size),
                zero_copy_pool_size=self.zero_copy_pool_size,
                dp_rank=self.dp_rank,
                dp_size=self.dp_size,
                mooncake_config=mooncake_config,
            ),
            prefetch_depth=self.prefetch_depth,
        )
