import argparse
from dataclasses import dataclass
from typing import Any, Dict, List

from specforge.modeling.target.sglang_backend import SGLangBackendArgs
__all__ = ["TrackerArgs", "SGLangBackendArgs"]


@dataclass
class TrackerArgs:
    report_to: str = "none"
    wandb_project: str = None
    wandb_name: str = None
    wandb_key: str = None
    swanlab_project: str = None
    swanlab_name: str = None
    swanlab_key: str = None
    mlflow_experiment_id: str = None
    mlflow_run_name: str = None
    mlflow_run_id: str = None
    mlflow_tracking_uri: str = None
    mlflow_registry_uri: str = None

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--report-to",
            type=str,
            default="none",
            choices=["wandb", "tensorboard", "swanlab", "mlflow", "none"],
            help="The integration to report results and logs to.",
        )
        # wandb-specific args
        parser.add_argument("--wandb-project", type=str, default=None)
        parser.add_argument("--wandb-name", type=str, default=None)
        parser.add_argument("--wandb-key", type=str, default=None, help="W&B API key.")
        # swanlab-specific args
        parser.add_argument(
            "--swanlab-project",
            type=str,
            default=None,
            help="The project name for swanlab.",
        )
        parser.add_argument(
            "--swanlab-name",
            type=str,
            default=None,
            help="The experiment name for swanlab.",
        )
        parser.add_argument(
            "--swanlab-key",
            type=str,
            default=None,
            help="The API key for swanlab non-interactive login.",
        )
        # mlflow-specific args
        parser.add_argument(
            "--mlflow-tracking-uri",
            type=str,
            default=None,
            help="The MLflow tracking URI. If not set, uses MLFLOW_TRACKING_URI environment variable or defaults to local './mlruns'.",
        )
        parser.add_argument(
            "--mlflow-experiment-name",
            type=str,
            default=None,
            help="The MLflow experiment name. If not set, uses MLFLOW_EXPERIMENT_NAME environment variable.",
        )
        parser.add_argument(
            "--mlflow-run-name",
            type=str,
            default=None,
            help="The MLflow run name. If not set, MLflow will auto-generate one.",
        )
