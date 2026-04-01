import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sglang.srt.server_args import ATTENTION_BACKEND_CHOICES


@dataclass
class TrackerArgs:
    report_to: str = "none"
    wandb_project: str = None
    wandb_name: str = None
    wandb_key: str = None
    wandb_offline: bool = False
    wandb_dir: str = None
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
        parser.add_argument(
            "--wandb-offline",
            action="store_true",
            help="Enable W&B offline mode and store logs locally.",
        )
        parser.add_argument(
            "--wandb-dir",
            type=str,
            default=None,
            help="Directory to store W&B files. Defaults to './wandb' under the project root when using W&B.",
        )
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


@dataclass
class SGLangBackendArgs:
    sglang_attention_backend: str = "fa3"
    sglang_mem_fraction_static: float = 0.4
    sglang_context_length: int = None
    sglang_enable_nccl_nvls: bool = False
    sglang_enable_symm_mem: bool = False
    sglang_enable_torch_compile: bool = True
    sglang_enable_dp_attention: bool = False
    sglang_enable_dp_lm_head: bool = False
    sglang_enable_piecewise_cuda_graph: bool = False
    sglang_piecewise_cuda_graph_max_tokens: int = 4096
    sglang_piecewise_cuda_graph_tokens: List[int] = None
    sglang_ep_size: int = 1
    sglang_max_running_requests: int = None  # assign based on batch size
    sglang_max_total_tokens: int = None  # assign based on batch size and seq length

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        # sglang arguments
        parser.add_argument(
            "--sglang-attention-backend",
            type=str,
            default="flashinfer",
            choices=ATTENTION_BACKEND_CHOICES,
            help="The attention backend of SGLang backend",
        )
        parser.add_argument(
            "--sglang-mem-fraction-static",
            type=float,
            default=0.4,
            help="The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors.",
        )
        parser.add_argument(
            "--sglang-context-length",
            type=int,
            default=None,
            help="The context length of the SGLang backend",
        )
        parser.add_argument(
            "--sglang-enable-nccl-nvls",
            action="store_true",
            help="Enable NCCL NVLS for prefill heavy requests when available for SGLang backend",
        )
        parser.add_argument(
            "--sglang-enable-symm-mem",
            action="store_true",
            help="Enable NCCL symmetric memory for fast collectives for SGLang backend",
        )
        parser.add_argument(
            "--sglang-enable-torch-compile",
            action="store_true",
            help="Optimize the model with torch.compile for SGLang backend",
        )
        parser.add_argument(
            "--sglang-enable-dp-attention",
            action="store_true",
            help="Enable DP attention for SGLang backend",
        )
        parser.add_argument(
            "--sglang-enable-dp-lm-head",
            action="store_true",
            help="Enable piecewise CUDA graph for SGLang backend",
        )
        parser.add_argument(
            "--sglang-enable-piecewise-cuda-graph",
            action="store_true",
            help="Enable piecewise CUDA graph for SGLang backend's prefill",
        )
        parser.add_argument(
            "--sglang-piecewise-cuda-graph-max-tokens",
            type=int,
            default=4096,
            help="Set the max tokens for piecewise CUDA graph for SGLang backend",
        )
        parser.add_argument(
            "--sglang-piecewise-cuda-graph-tokens",
            type=int,
            nargs="+",
            default=None,
            help="Set the list of tokens when using piecewise cuda graph for SGLang backend",
        )
        parser.add_argument(
            "--sglang-ep-size",
            type=int,
            default=1,
            help="The ep size of the SGLang backend",
        )

    @staticmethod
    def from_args(args: argparse.Namespace) -> "SGLangBackendArgs":
        return SGLangBackendArgs(
            sglang_attention_backend=args.sglang_attention_backend,
            sglang_mem_fraction_static=args.sglang_mem_fraction_static,
            sglang_context_length=args.sglang_context_length,
            sglang_enable_nccl_nvls=args.sglang_enable_nccl_nvls,
            sglang_enable_symm_mem=args.sglang_enable_symm_mem,
            sglang_enable_torch_compile=args.sglang_enable_torch_compile,
            sglang_enable_dp_attention=args.sglang_enable_dp_attention,
            sglang_enable_dp_lm_head=args.sglang_enable_dp_lm_head,
            sglang_enable_piecewise_cuda_graph=args.sglang_enable_piecewise_cuda_graph,
            sglang_piecewise_cuda_graph_max_tokens=args.sglang_piecewise_cuda_graph_max_tokens,
            sglang_piecewise_cuda_graph_tokens=args.sglang_piecewise_cuda_graph_tokens,
            sglang_ep_size=args.sglang_ep_size,
            sglang_max_running_requests=(
                args.target_batch_size if hasattr(args, "target_batch_size") else None
            ),
            sglang_max_total_tokens=(
                args.target_batch_size * args.max_length
                if hasattr(args, "target_batch_size") and hasattr(args, "max_length")
                else None
            ),
        )

    def to_kwargs(self) -> Dict[str, Any]:
        return dict(
            attention_backend=self.sglang_attention_backend,
            mem_fraction_static=self.sglang_mem_fraction_static,
            context_length=self.sglang_context_length,
            enable_nccl_nvls=self.sglang_enable_nccl_nvls,
            enable_symm_mem=self.sglang_enable_symm_mem,
            enable_torch_compile=self.sglang_enable_torch_compile,
            enable_dp_attention=self.sglang_enable_dp_attention,
            enable_dp_lm_head=self.sglang_enable_dp_lm_head,
            enable_piecewise_cuda_graph=self.sglang_enable_piecewise_cuda_graph,
            piecewise_cuda_graph_max_tokens=self.sglang_piecewise_cuda_graph_max_tokens,
            piecewise_cuda_graph_tokens=self.sglang_piecewise_cuda_graph_tokens,
            ep_size=self.sglang_ep_size,
            max_running_requests=self.sglang_max_running_requests,
            max_total_tokens=self.sglang_max_total_tokens,
        )


@dataclass
class RayArgs:
    """Parameters for Ray cluster initialisation."""

    ray_address: Optional[str] = None
    ray_num_gpus: Optional[int] = None
    ray_namespace: str = "specforge"

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--ray-address",
            type=str,
            default=None,
            help=(
                "Ray cluster address (e.g. 'auto' or 'ray://head:10001'). "
                "If not set, Ray will start a local cluster."
            ),
        )
        parser.add_argument(
            "--ray-num-gpus",
            type=int,
            default=None,
            help=(
                "Total GPUs to expose to the local Ray cluster. "
                "Only used when --ray-address is not set."
            ),
        )
        parser.add_argument(
            "--ray-namespace",
            type=str,
            default="specforge",
            help="Ray namespace used for actor naming isolation.",
        )

    @staticmethod
    def from_args(args: argparse.Namespace) -> "RayArgs":
        return RayArgs(
            ray_address=args.ray_address,
            ray_num_gpus=args.ray_num_gpus,
            ray_namespace=args.ray_namespace,
        )


@dataclass
class DisaggregateArgs:
    """Controls whether inference and training run on separate GPUs."""

    disaggregate: bool = False
    rollout_num_gpus: Optional[int] = None
    train_num_gpus: Optional[int] = None
    rollout_tp_size: int = 1
    train_tp_size: int = 1
    train_sp_ulysses_size: int = 1
    train_sp_ring_size: int = 1
    transfer_backend: str = "ray"
    rollout_async: bool = False

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--disaggregate",
            action="store_true",
            help=(
                "Enable disaggregated (inference/training separated) mode. "
                "When set, rollout GPUs and train GPUs are isolated."
            ),
        )
        parser.add_argument(
            "--rollout-num-gpus",
            type=int,
            default=None,
            help="Number of GPUs dedicated to target-model rollout (disaggregated mode).",
        )
        parser.add_argument(
            "--train-num-gpus",
            type=int,
            default=None,
            help=(
                "Number of GPUs dedicated to draft-model training. "
                "Must equal dp_size * train_tp_size."
            ),
        )
        parser.add_argument(
            "--rollout-tp-size",
            type=int,
            default=1,
            help="Tensor-parallel degree for the rollout (target) model.",
        )
        parser.add_argument(
            "--train-tp-size",
            type=int,
            default=1,
            help="Tensor-parallel degree for the draft model during training.",
        )
        parser.add_argument(
            "--train-sp-ulysses-size",
            type=int,
            default=1,
            help="Ulysses sequence-parallel degree for training (requires --attention-backend usp).",
        )
        parser.add_argument(
            "--train-sp-ring-size",
            type=int,
            default=1,
            help="Ring sequence-parallel degree for training (requires --attention-backend usp).",
        )
        parser.add_argument(
            "--transfer-backend",
            type=str,
            default="ray",
            choices=["ray"],
            help="Backend used to transfer rollout tensors from rollout to train workers.",
        )
        parser.add_argument(
            "--rollout-async",
            action="store_true",
            help=(
                "Prefetch the next rollout batch while the current training step runs "
                "(advanced pipeline parallelism)."
            ),
        )

    @staticmethod
    def from_args(args: argparse.Namespace) -> "DisaggregateArgs":
        return DisaggregateArgs(
            disaggregate=args.disaggregate,
            rollout_num_gpus=args.rollout_num_gpus,
            train_num_gpus=args.train_num_gpus,
            rollout_tp_size=args.rollout_tp_size,
            train_tp_size=args.train_tp_size,
            train_sp_ulysses_size=args.train_sp_ulysses_size,
            train_sp_ring_size=args.train_sp_ring_size,
            transfer_backend=args.transfer_backend,
            rollout_async=args.rollout_async,
        )

    def validate(self, args: argparse.Namespace) -> None:
        """
        Validate disaggregate-related argument combinations.

        Raises
        ------
        ValueError if any constraint is violated.
        """
        if self.disaggregate:
            if self.rollout_num_gpus is None:
                raise ValueError("--rollout-num-gpus is required in disaggregated mode.")
            if self.train_num_gpus is None:
                raise ValueError("--train-num-gpus is required in disaggregated mode.")
            if self.rollout_num_gpus % self.rollout_tp_size != 0:
                raise ValueError(
                    f"rollout_num_gpus ({self.rollout_num_gpus}) must be divisible by "
                    f"rollout_tp_size ({self.rollout_tp_size})."
                )
            if self.train_num_gpus % self.train_tp_size != 0:
                raise ValueError(
                    f"train_num_gpus ({self.train_num_gpus}) must be divisible by "
                    f"train_tp_size ({self.train_tp_size})."
                )
            sp_size = self.train_sp_ulysses_size * self.train_sp_ring_size
            if self.train_num_gpus % sp_size != 0:
                raise ValueError(
                    f"train_num_gpus ({self.train_num_gpus}) must be divisible by "
                    f"train_sp_size ({sp_size})."
                )

        sp_size = self.train_sp_ulysses_size * self.train_sp_ring_size
        if sp_size > 1:
            if getattr(args, "attention_backend", "sdpa") != "usp":
                raise ValueError(
                    "SP (train_sp_ulysses_size * train_sp_ring_size > 1) "
                    "requires --attention-backend usp."
                )
            if getattr(args, "batch_size", 1) != 1:
                raise ValueError(
                    "SP (train_sp_ulysses_size * train_sp_ring_size > 1) "
                    "requires --batch-size 1."
                )
