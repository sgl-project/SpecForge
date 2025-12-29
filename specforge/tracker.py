# tracker.py

import abc
import netrc
import os
from typing import Any, Dict, Optional

import torch.distributed as dist

# --- Lazy Imports ---
# These libraries are imported only when their respective trackers are used.
try:
    import wandb
except ImportError:
    wandb = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    import swanlab
except ImportError:
    swanlab = None

try:
    import mlflow
except ImportError:
    mlflow = None


# --- End Lazy Imports ---


class Tracker(abc.ABC):
    """
    Abstract Base Class for experiment trackers.

    Each tracker implementation should handle its own initialization, logging,
    and cleanup. It should also provide a class method to validate
    command-line arguments before initialization.
    """

    def __init__(self, args, output_dir: str):
        self.args = args
        self.output_dir = output_dir
        self.rank = dist.get_rank()
        self.is_initialized = False

    @classmethod
    @abc.abstractmethod
    def validate_args(cls, parser, args) -> None:
        """
        Validate necessary arguments for this tracker.
        This method is called during argument parsing.
        It should raise an error if required arguments are missing.
        """

    @abc.abstractmethod
    def log(self, log_dict: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to the tracker.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """
        Close the tracker and clean up resources.
        """


class NoOpTracker(Tracker):
    """A tracker that does nothing, for when no tracking is desired."""

    @classmethod
    def validate_args(cls, parser, args):
        pass  # No arguments to validate

    def __init__(self, args, output_dir: str):
        super().__init__(args, output_dir)
        self.is_initialized = True  # Considered initialized to do nothing

    def log(self, log_dict: Dict[str, Any], step: Optional[int] = None):
        pass  # Do nothing

    def close(self):
        pass  # Do nothing


class WandbTracker(Tracker):
    """Tracks experiments using Weights & Biases."""

    @classmethod
    def validate_args(cls, parser, args):
        if wandb is None:
            parser.error(
                "To use --report-to wandb, you must install wandb: 'pip install wandb'"
            )

        if args.wandb_key is not None:
            return

        if "WANDB_API_KEY" in os.environ:
            args.wandb_key = os.environ["WANDB_API_KEY"]
            return

        try:
            netrc_path = os.path.expanduser("~/.netrc")
            if os.path.exists(netrc_path):
                netrc_file = netrc.netrc(netrc_path)
                if "api.wandb.ai" in netrc_file.hosts:
                    _, _, password = netrc_file.authenticators("api.wandb.ai")
                    if password:
                        args.wandb_key = password
                        return
        except (FileNotFoundError, netrc.NetrcParseError):
            pass

        if args.wandb_key is None:
            parser.error(
                "When --report-to is 'wandb', you must provide a wandb API key via one of:\n"
                "  1. --wandb-key argument\n"
                "  2. WANDB_API_KEY environment variable\n"
                "  3. `wandb login` command"
            )

    def __init__(self, args, output_dir: str):
        super().__init__(args, output_dir)
        if self.rank == 0:
            wandb.login(key=args.wandb_key)
            wandb.init(
                project=args.wandb_project, name=args.wandb_name, config=vars(args)
            )
            self.is_initialized = True

    def log(self, log_dict: Dict[str, Any], step: Optional[int] = None):
        if self.rank == 0 and self.is_initialized:
            wandb.log(log_dict, step=step)

    def close(self):
        if self.rank == 0 and self.is_initialized and wandb.run:
            wandb.finish()
            self.is_initialized = False


class SwanlabTracker(Tracker):
    """Tracks experiments using SwanLab."""

    @classmethod
    def validate_args(cls, parser, args):
        if swanlab is None:
            parser.error(
                "To use --report-to swanlab, you must install swanlab: 'pip install swanlab'"
            )

        if args.swanlab_key is not None:
            return
        if "SWANLAB_API_KEY" in os.environ:
            args.swanlab_key = os.environ["SWANLAB_API_KEY"]
            return
        # Swanlab can run in anonymous mode if no key is provided in a non-distributed env.
        # However, a key is often required for distributed runs to sync correctly.
        if (
            dist.is_initialized()
            and dist.get_world_size() > 1
            and args.swanlab_key is None
        ):
            parser.error(
                "In a distributed environment, when --report-to is 'swanlab', you must provide a swanlab API key via:\n"
                "  1. --swanlab-key argument\n"
                "  2. SWANLAB_API_KEY environment variable"
            )

    def __init__(self, args, output_dir: str):
        super().__init__(args, output_dir)
        if self.rank == 0:
            if args.swanlab_key:
                swanlab.login(api_key=args.swanlab_key)

            swanlog_dir = os.path.join(output_dir, "swanlog")
            os.makedirs(swanlog_dir, exist_ok=True)
            swanlab.init(
                project=args.swanlab_project,
                experiment_name=args.swanlab_name,
                config=vars(args),
                logdir=swanlog_dir,
            )
            self.is_initialized = True

    def log(self, log_dict: Dict[str, Any], step: Optional[int] = None):
        if self.rank == 0 and self.is_initialized:
            swanlab.log(log_dict, step=step)

    def close(self):
        if self.rank == 0 and self.is_initialized and swanlab.get_run() is not None:
            swanlab.finish()
            self.is_initialized = False


class TensorboardTracker(Tracker):
    """Tracks experiments using TensorBoard."""

    @classmethod
    def validate_args(cls, parser, args):
        if SummaryWriter is None:
            parser.error(
                "To use --report-to tensorboard, you must have tensorboard installed: 'pip install tensorboard'"
            )

    def __init__(self, args, output_dir: str):
        super().__init__(args, output_dir)
        if self.rank == 0:
            log_dir = os.path.join(output_dir, "runs")
            self.writer = SummaryWriter(log_dir=log_dir)
            self.is_initialized = True

    def log(self, log_dict: Dict[str, Any], step: Optional[int] = None):
        if self.rank == 0 and self.is_initialized:
            for key, value in log_dict.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, global_step=step)

    def close(self):
        if self.rank == 0 and self.is_initialized:
            self.writer.close()
            self.is_initialized = False


class MLflowTracker(Tracker):
    """Tracks experiments using MLflow."""

    @classmethod
    def validate_args(cls, parser, args):
        if mlflow is None:
            parser.error(
                "To use --report-to mlflow, you must install mlflow: 'pip install mlflow'"
            )
        # Set tracking URI from environment variable if not explicitly provided
        if args.mlflow_tracking_uri is None and "MLFLOW_TRACKING_URI" in os.environ:
            args.mlflow_tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        elif args.mlflow_tracking_uri is None:
            print(
                "Warning: MLflow tracking URI not set. Defaulting to local './mlruns'."
            )

        # Set experiment name from environment variable if not explicitly provided
        if (
            args.mlflow_experiment_name is None
            and "MLFLOW_EXPERIMENT_NAME" in os.environ
        ):
            args.mlflow_experiment_name = os.environ["MLFLOW_EXPERIMENT_NAME"]

    def __init__(self, args, output_dir: str):
        super().__init__(args, output_dir)
        if self.rank == 0:
            if args.mlflow_tracking_uri:
                mlflow.set_tracking_uri(args.mlflow_tracking_uri)

            # This will either use the set URI or the default
            mlflow.set_experiment(args.mlflow_experiment_name)
            mlflow.start_run(run_name=args.mlflow_run_name)
            mlflow.log_params(vars(args))
            self.is_initialized = True

    def log(self, log_dict: Dict[str, Any], step: Optional[int] = None):
        if self.rank == 0 and self.is_initialized:
            # MLflow's log_metrics takes a dictionary directly
            mlflow.log_metrics(log_dict, step=step)

    def close(self):
        if self.rank == 0 and self.is_initialized:
            mlflow.end_run()
            self.is_initialized = False


# --- Tracker Factory ---
TRACKER_REGISTRY = {
    "wandb": WandbTracker,
    "swanlab": SwanlabTracker,
    "tensorboard": TensorboardTracker,
    "mlflow": MLflowTracker,
    "none": NoOpTracker,
}


def get_tracker_class(report_to: str) -> Optional[Tracker]:
    """Returns the tracker class based on the name."""
    return TRACKER_REGISTRY.get(report_to)


def create_tracker(args, output_dir: str) -> Tracker:
    """Factory function to create an experiment tracker instance."""
    tracker_class = get_tracker_class(args.report_to)
    if not tracker_class:
        raise ValueError(f"Unsupported report_to type: {args.report_to}")
    return tracker_class(args, output_dir)
