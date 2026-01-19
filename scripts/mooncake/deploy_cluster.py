#!/usr/bin/env python3
"""
Mooncake Store cluster deployment helper.

This script helps deploy Mooncake Store components across multiple nodes
for distributed Eagle3 training.

Usage:
    python deploy_cluster.py --config cluster_config.json
    python deploy_cluster.py --generate-config > cluster_config.json
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class NodeConfig:
    """Configuration for a single node."""

    hostname: str
    role: str
    ssh_user: str = "root"
    ssh_port: int = 22
    ssh_key: Optional[str] = None
    gpu_ids: List[int] = field(default_factory=list)
    extra_env: Dict[str, str] = field(default_factory=dict)


@dataclass
class ClusterConfig:
    """Configuration for the entire cluster."""

    master_node: str
    master_port: int = 50051
    metadata_port: int = 8080
    task_queue_port: int = 5555
    notify_port: int = 5556
    protocol: str = "tcp"
    global_segment_size: str = "4GB"
    local_buffer_size: str = "512MB"
    nodes: List[NodeConfig] = field(default_factory=list)


def generate_sample_config() -> dict:
    """Generate a sample cluster configuration."""
    return {
        "master_node": "master-host",
        "master_port": 50051,
        "metadata_port": 8080,
        "task_queue_port": 5555,
        "notify_port": 5556,
        "protocol": "tcp",
        "global_segment_size": "4GB",
        "local_buffer_size": "512MB",
        "nodes": [
            {
                "hostname": "master-host",
                "role": "master",
                "ssh_user": "root",
                "ssh_port": 22,
            },
            {
                "hostname": "inference-host-1",
                "role": "inference",
                "ssh_user": "root",
                "gpu_ids": [0, 1, 2, 3],
                "extra_env": {"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            },
            {
                "hostname": "inference-host-2",
                "role": "inference",
                "ssh_user": "root",
                "gpu_ids": [0, 1, 2, 3],
            },
            {
                "hostname": "training-host-1",
                "role": "training",
                "ssh_user": "root",
                "gpu_ids": [0, 1, 2, 3, 4, 5, 6, 7],
            },
        ],
    }


def generate_worker_config(
    cluster_config: dict, node: dict, model_path: str
) -> dict:
    """Generate inference worker configuration for a node."""
    master_addr = f"{cluster_config['master_node']}:{cluster_config['master_port']}"
    task_queue_addr = f"tcp://{cluster_config['master_node']}:{cluster_config['task_queue_port']}"
    notify_addr = f"tcp://{cluster_config['master_node']}:{cluster_config['notify_port']}"

    return {
        "model_path": model_path,
        "task_queue_addr": task_queue_addr,
        "notify_addr": notify_addr,
        "tp_size": len(node.get("gpu_ids", [1])) or 1,
        "mooncake": {
            "local_hostname": node["hostname"],
            "master_server_address": master_addr,
            "metadata_port": cluster_config["metadata_port"],
            "global_segment_size": cluster_config["global_segment_size"],
            "local_buffer_size": cluster_config["local_buffer_size"],
            "protocol": cluster_config["protocol"],
        },
    }


def generate_training_env(cluster_config: dict, node: dict) -> dict:
    """Generate environment variables for training nodes."""
    master_addr = f"{cluster_config['master_node']}:{cluster_config['master_port']}"
    task_queue_addr = f"tcp://{cluster_config['master_node']}:{cluster_config['task_queue_port']}"
    notify_addr = f"tcp://{cluster_config['master_node']}:{cluster_config['notify_port']}"

    return {
        "MOONCAKE_MASTER_SERVER": master_addr,
        "MOONCAKE_MASTER_HOST": cluster_config["master_node"],
        "MOONCAKE_MASTER_PORT": str(cluster_config["master_port"]),
        "MOONCAKE_METADATA_PORT": str(cluster_config["metadata_port"]),
        "TASK_QUEUE_ADDR": task_queue_addr,
        "NOTIFY_ADDR": notify_addr,
        "MOONCAKE_LOCAL_HOSTNAME": node["hostname"],
        "MOONCAKE_PROTOCOL": cluster_config["protocol"],
        "MOONCAKE_GLOBAL_SEGMENT_SIZE": cluster_config["global_segment_size"],
        "MOONCAKE_LOCAL_BUFFER_SIZE": cluster_config["local_buffer_size"],
    }


def run_ssh_command(
    node: dict, command: str, check: bool = True
) -> subprocess.CompletedProcess:
    """Run a command on a remote node via SSH."""
    ssh_args = ["ssh"]

    if node.get("ssh_key"):
        ssh_args.extend(["-i", node["ssh_key"]])

    ssh_args.extend(["-p", str(node.get("ssh_port", 22))])
    ssh_args.append(f"{node.get('ssh_user', 'root')}@{node['hostname']}")
    ssh_args.append(command)

    return subprocess.run(ssh_args, check=check, capture_output=True, text=True)


def start_master(cluster_config: dict, node: dict, script_dir: str) -> None:
    """Start Mooncake Master Service on the master node."""
    print(f"Starting Master Service on {node['hostname']}...")

    cmd = f"""
    cd {script_dir} && \
    PORT={cluster_config['master_port']} \
    HTTP_METADATA_PORT={cluster_config['metadata_port']} \
    ENABLE_HTTP_METADATA=true \
    nohup ./start_master.sh > /tmp/mooncake_master.log 2>&1 &
    """

    try:
        run_ssh_command(node, cmd)
        print(f"  Master Service started on {node['hostname']}")
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Failed to start Master Service: {e.stderr}")


def start_inference_worker(
    cluster_config: dict, node: dict, model_path: str, worker_script: str
) -> None:
    """Start inference worker on a node."""
    print(f"Starting inference worker on {node['hostname']}...")

    config = generate_worker_config(cluster_config, node, model_path)
    config_json = json.dumps(config)

    env_vars = " ".join(
        f"{k}={v}" for k, v in node.get("extra_env", {}).items()
    )

    cmd = f"""
    {env_vars} python {worker_script} --config '{config_json}' \
    > /tmp/inference_worker.log 2>&1 &
    """

    try:
        run_ssh_command(node, cmd)
        print(f"  Inference worker started on {node['hostname']}")
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Failed to start inference worker: {e.stderr}")


def print_startup_instructions(cluster_config: dict) -> None:
    """Print manual startup instructions."""
    master_addr = f"{cluster_config['master_node']}:{cluster_config['master_port']}"
    task_queue_addr = f"tcp://{cluster_config['master_node']}:{cluster_config['task_queue_port']}"
    notify_addr = f"tcp://{cluster_config['master_node']}:{cluster_config['notify_port']}"

    print("\n" + "=" * 60)
    print("CLUSTER STARTUP INSTRUCTIONS")
    print("=" * 60)

    print("\n1. Start Mooncake Master Service (on master node):")
    print(f"   cd scripts/mooncake && ./start_master.sh --port {cluster_config['master_port']}")
    print(f"   # This also starts the built-in HTTP metadata server on port {cluster_config['metadata_port']}")

    print("\n2. Start Inference Workers (on each inference node):")
    print(f"   python scripts/run_inference_worker.py \\")
    print(f"       --model-path <MODEL_PATH> \\")
    print(f"       --task-queue-addr {task_queue_addr} \\")
    print(f"       --notify-addr {notify_addr} \\")
    print(f"       --mooncake-master-addr {master_addr}")
    print(f"   # Metadata server URL is automatically derived from master address")

    print("\n3. Run Training (on training nodes):")
    print(f"   python scripts/train_eagle3.py \\")
    print(f"       --target-model-backend remote \\")
    print(f"       --task-queue-addr {task_queue_addr} \\")
    print(f"       --notify-addr {notify_addr} \\")
    print(f"       --mooncake-master-addr {master_addr} \\")
    print(f"       ... (other training args)")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Mooncake cluster deployment helper")
    parser.add_argument(
        "--config", type=str, help="Path to cluster configuration JSON file"
    )
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Generate sample configuration",
    )
    parser.add_argument(
        "--model-path", type=str, help="Model path for inference workers"
    )
    parser.add_argument(
        "--print-instructions",
        action="store_true",
        help="Print startup instructions without deploying",
    )
    parser.add_argument(
        "--deploy", action="store_true", help="Actually deploy to nodes via SSH"
    )

    args = parser.parse_args()

    if args.generate_config:
        print(json.dumps(generate_sample_config(), indent=2))
        return

    if not args.config:
        parser.error("--config is required (or use --generate-config)")

    with open(args.config) as f:
        cluster_config = json.load(f)

    if args.print_instructions or not args.deploy:
        print_startup_instructions(cluster_config)
        if not args.deploy:
            print("\nUse --deploy to actually deploy to nodes via SSH")
        return

    if args.deploy:
        if not args.model_path:
            parser.error("--model-path is required for deployment")

        script_dir = os.path.dirname(os.path.abspath(__file__))

        for node in cluster_config["nodes"]:
            if node["role"] == "master":
                start_master(cluster_config, node, script_dir)
            elif node["role"] == "inference":
                worker_script = os.path.join(
                    os.path.dirname(script_dir), "run_inference_worker.py"
                )
                start_inference_worker(
                    cluster_config, node, args.model_path, worker_script
                )

        print("\nDeployment complete!")
        print_startup_instructions(cluster_config)


if __name__ == "__main__":
    main()
