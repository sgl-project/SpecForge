import argparse
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List

from sglang.srt.server_args import (
    ATTENTION_BACKEND_CHOICES,
    MOE_A2A_BACKEND_CHOICES,
    MOE_RUNNER_BACKEND_CHOICES,
    ServerArgs,
)


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
    sglang_kv_cache_dtype: str = "auto"
    sglang_reasoning_parser: str = None
    sglang_tool_call_parser: str = None
    sglang_enable_nccl_nvls: bool = False
    sglang_enable_symm_mem: bool = False
    sglang_enable_torch_compile: bool = True
    sglang_enable_dp_attention: bool = False
    sglang_enable_dp_lm_head: bool = False
    sglang_enable_piecewise_cuda_graph: bool = False
    sglang_piecewise_cuda_graph_max_tokens: int = 4096
    sglang_piecewise_cuda_graph_tokens: List[int] = None
    sglang_ep_size: int = 1
    sglang_dp_size: int = 1
    sglang_moe_a2a_backend: str = "none"
    sglang_moe_runner_backend: str = "auto"
    sglang_max_running_requests: int = None  # assign based on batch size
    sglang_max_total_tokens: int = None  # assign based on batch size and seq length
    sglang_enforce_disable_flashinfer_allreduce_fusion: bool = False
    sglang_disable_flashinfer_autotune: bool = False

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
            "--sglang-kv-cache-dtype",
            type=str,
            default="auto",
            help="KV cache dtype for the SGLang backend, e.g. auto, bfloat16, fp8_e4m3, or fp4_e2m1.",
        )
        parser.add_argument(
            "--sglang-reasoning-parser",
            type=str,
            default=None,
            help="Reasoning parser name for SGLang model config compatibility.",
        )
        parser.add_argument(
            "--sglang-tool-call-parser",
            type=str,
            default=None,
            help="Tool-call parser name for SGLang model config compatibility.",
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
        parser.add_argument(
            "--sglang-dp-size",
            type=int,
            default=1,
            help="The data-parallel size of the SGLang backend. With "
            "--sglang-enable-dp-attention, set this to the world/tp size to make "
            "attention data-parallel (attn_tp = tp_size // dp_size).",
        )
        parser.add_argument(
            "--sglang-moe-a2a-backend",
            type=str,
            default="none",
            choices=MOE_A2A_BACKEND_CHOICES,
            help="SGLang MoE A2A backend.",
        )
        parser.add_argument(
            "--sglang-moe-runner-backend",
            type=str,
            default="auto",
            choices=MOE_RUNNER_BACKEND_CHOICES,
            help="SGLang MoE runner backend.",
        )
        parser.add_argument(
            "--sglang-max-running-requests",
            type=int,
            default=None,
            help="Cap SGLang concurrent requests for the target backend.",
        )
        parser.add_argument(
            "--sglang-max-total-tokens",
            type=int,
            default=None,
            help="Cap SGLang KV-cache tokens for the target backend.",
        )
        parser.add_argument(
            "--sglang-enforce-disable-flashinfer-allreduce-fusion",
            action="store_true",
            help="Force-disable SGLang FlashInfer all-reduce fusion.",
        )
        parser.add_argument(
            "--sglang-disable-flashinfer-autotune",
            action="store_true",
            help="Disable SGLang FlashInfer autotune during target backend init.",
        )

    @staticmethod
    def from_args(args: argparse.Namespace) -> "SGLangBackendArgs":
        return SGLangBackendArgs(
            sglang_attention_backend=args.sglang_attention_backend,
            sglang_mem_fraction_static=args.sglang_mem_fraction_static,
            sglang_context_length=args.sglang_context_length,
            sglang_kv_cache_dtype=args.sglang_kv_cache_dtype,
            sglang_reasoning_parser=args.sglang_reasoning_parser,
            sglang_tool_call_parser=args.sglang_tool_call_parser,
            sglang_enable_nccl_nvls=args.sglang_enable_nccl_nvls,
            sglang_enable_symm_mem=args.sglang_enable_symm_mem,
            sglang_enable_torch_compile=args.sglang_enable_torch_compile,
            sglang_enable_dp_attention=args.sglang_enable_dp_attention,
            sglang_enable_dp_lm_head=args.sglang_enable_dp_lm_head,
            sglang_enable_piecewise_cuda_graph=args.sglang_enable_piecewise_cuda_graph,
            sglang_piecewise_cuda_graph_max_tokens=args.sglang_piecewise_cuda_graph_max_tokens,
            sglang_piecewise_cuda_graph_tokens=args.sglang_piecewise_cuda_graph_tokens,
            sglang_ep_size=args.sglang_ep_size,
            sglang_dp_size=args.sglang_dp_size,
            sglang_moe_a2a_backend=args.sglang_moe_a2a_backend,
            sglang_moe_runner_backend=args.sglang_moe_runner_backend,
            sglang_max_running_requests=(
                args.sglang_max_running_requests
                if args.sglang_max_running_requests is not None
                else args.target_batch_size
                if hasattr(args, "target_batch_size")
                else None
            ),
            sglang_max_total_tokens=(
                args.sglang_max_total_tokens
                if args.sglang_max_total_tokens is not None
                else args.target_batch_size * args.max_length
                if hasattr(args, "target_batch_size") and hasattr(args, "max_length")
                else None
            ),
            sglang_enforce_disable_flashinfer_allreduce_fusion=(
                args.sglang_enforce_disable_flashinfer_allreduce_fusion
            ),
            sglang_disable_flashinfer_autotune=args.sglang_disable_flashinfer_autotune,
        )

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs = dict(
            attention_backend=self.sglang_attention_backend,
            mem_fraction_static=self.sglang_mem_fraction_static,
            context_length=self.sglang_context_length,
            kv_cache_dtype=self.sglang_kv_cache_dtype,
            reasoning_parser=self.sglang_reasoning_parser,
            tool_call_parser=self.sglang_tool_call_parser,
            enable_nccl_nvls=self.sglang_enable_nccl_nvls,
            enable_symm_mem=self.sglang_enable_symm_mem,
            enable_torch_compile=self.sglang_enable_torch_compile,
            enable_dp_attention=self.sglang_enable_dp_attention,
            enable_dp_lm_head=self.sglang_enable_dp_lm_head,
            enable_piecewise_cuda_graph=self.sglang_enable_piecewise_cuda_graph,
            piecewise_cuda_graph_max_tokens=self.sglang_piecewise_cuda_graph_max_tokens,
            piecewise_cuda_graph_tokens=self.sglang_piecewise_cuda_graph_tokens,
            ep_size=self.sglang_ep_size,
            dp_size=self.sglang_dp_size,
            moe_a2a_backend=self.sglang_moe_a2a_backend,
            moe_runner_backend=self.sglang_moe_runner_backend,
            max_running_requests=self.sglang_max_running_requests,
            max_total_tokens=self.sglang_max_total_tokens,
            enforce_disable_flashinfer_allreduce_fusion=(
                self.sglang_enforce_disable_flashinfer_allreduce_fusion
            ),
            disable_flashinfer_autotune=self.sglang_disable_flashinfer_autotune,
        )
        parameters = inspect.signature(ServerArgs).parameters
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            return kwargs
        return {key: value for key, value in kwargs.items() if key in parameters}
