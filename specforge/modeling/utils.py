import json
import os
from typing import Optional

import torch


@torch.no_grad()
def padding(tensor, left=True):
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    else:
        tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor


def get_ib_devices_for_gpu(ib_device_str: Optional[str], gpu_id: int) -> Optional[str]:
    """
    Parse IB device string and get IB devices for a specific GPU ID.

    Supports two formats:
    1. Simple format: "mlx5_0,mlx5_1" - same devices for all GPUs
    2. JSON file: path to a JSON file with GPU-to-devices mapping
       Example JSON: {"0": "mlx5_0,mlx5_1", "1": "mlx5_2,mlx5_3"}

    Args:
        ib_device_str: Device string or path to JSON file
        gpu_id: The GPU ID to get devices for

    Returns:
        IB devices string for the GPU, or None if not available
    """
    if ib_device_str is None or not ib_device_str.strip():
        return None

    ib_device_str = ib_device_str.strip()

    # Check if it's a JSON file
    if ib_device_str.endswith(".json") and os.path.isfile(ib_device_str):
        try:
            with open(ib_device_str, "r") as f:
                gpu_mapping = json.load(f)

            if not isinstance(gpu_mapping, dict):
                raise ValueError(
                    f"JSON file must contain a dictionary, got {type(gpu_mapping)}"
                )

            # Convert keys to integers and validate
            normalized_mapping = {}
            for gpu_key, ib_devices in gpu_mapping.items():
                if isinstance(gpu_key, str) and gpu_key.isdigit():
                    normalized_mapping[int(gpu_key)] = ib_devices.strip()
                elif isinstance(gpu_key, int):
                    normalized_mapping[gpu_key] = ib_devices.strip()
                else:
                    raise ValueError(f"Invalid GPU key: {gpu_key} (must be an integer)")

            if gpu_id not in normalized_mapping:
                raise ValueError(
                    f"No IB devices configured for GPU {gpu_id}. "
                    f"Available GPUs: {list(normalized_mapping.keys())}"
                )

            return normalized_mapping[gpu_id]

        except (IOError, OSError) as e:
            raise RuntimeError(f"Failed to read JSON file {ib_device_str}: {e}") from e
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"Failed to parse JSON file {ib_device_str}: {e}") from e

    # Simple format - return same devices for all GPUs
    return ib_device_str


def parse_mooncake_device_name(device_name_arg: str, worker_id: int) -> str:
    """
    Parse mooncake device name with automatic GPU ID detection.

    This function:
    1. Auto-detects GPU ID from CUDA_VISIBLE_DEVICES or falls back to worker_id
    2. Calls get_ib_devices_for_gpu to parse the device configuration
    3. Returns the resolved device name string

    Args:
        device_name_arg: Device name argument (simple string or JSON file path)
        worker_id: Worker ID to use as fallback for GPU ID

    Returns:
        Resolved device name string, or empty string if not available
    """
    import logging

    if not device_name_arg:
        return ""

    # Auto-detect GPU ID from environment variables or use worker_id
    gpu_id = None

    # Try to get GPU ID from CUDA_VISIBLE_DEVICES
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        try:
            devices = [
                int(d.strip())
                for d in cuda_visible_devices.split(",")
                if d.strip().isdigit()
            ]
            if devices:
                gpu_id = devices[0]
                logging.info(
                    f"Detected GPU ID {gpu_id} from CUDA_VISIBLE_DEVICES={cuda_visible_devices}"
                )
        except (ValueError, IndexError):
            pass

    # Fall back to worker_id if GPU ID not detected
    if gpu_id is None:
        gpu_id = worker_id
        logging.info(f"Using worker_id {gpu_id} as GPU ID for IB device mapping")

    # Parse device name
    device_name = get_ib_devices_for_gpu(device_name_arg, gpu_id)
    if device_name:
        logging.info(f"IB device name for GPU {gpu_id}: {device_name}")
        return device_name
    else:
        logging.warning(f"No IB device found for GPU {gpu_id}")
        return ""
