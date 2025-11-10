import argparse
import dataclasses
import os
from typing import Optional

import torch.distributed as dist

from .utils import get_last_checkpoint, print_on_rank0


@dataclasses.dataclass
class SpecForgeArgs:
    target_model_path: str
    draft_model_config: str
    train_data_path: str
    output_dir: str
    eval_data_path: Optional[str] = None

    embedding_key: str = "model.embed_tokens.weight"
    lm_head_key: str = "lm_head.weight"
    max_grad_norm: float = 0.5
    build_dataset_num_proc: int = 8
    target_model_backend: str = "sglang"
    target_tp_size: int = 1
    target_dp_size: int = 1
    draft_tp_size: int = 1
    draft_dp_size: int = 1
    draft_global_batch_size: int = 16
    draft_micro_batch_size: int = 1
    draft_accumulation_steps: int = 1  # auto calculated
    num_epochs: int = 10
    learning_rate: float = 5e-5
    log_interval: int = 1
    eval_interval: int = -1
    save_interval: int = -1
    dist_timeout: int = 10
    save_per_epoch: bool = True  # auto calculated
    total_steps: Optional[int] = None
    max_num_saved_checkpoints: int = -1
    seed: int = 0
    draft_attention_backend: str = "flex_attention"
    report_to: str = "none"
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_key: Optional[str] = None
    swanlab_project: Optional[str] = None
    swanlab_name: Optional[str] = None
    swanlab_key: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None
    mlflow_run_name: Optional[str] = None
    resume: bool = False
    profile_start_step: int = 30
    profile_num_steps: int = 4
    cache_dir: str = "./cache"
    ttt_length: int = 7
    is_vlm: bool = False
    is_preformatted: bool = False
    draft_model_last_checkpoint: str = None
    max_length: int = 2048
    warmup_ratio: float = 0.015
    chat_template: Optional[str] = None
    enable_zero2: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        # add model-related arguments
        parser.add_argument(
            "--target-model-path", "--model-path", type=str, required=True
        )
        parser.add_argument("--draft-model-config", type=str, required=True)
        parser.add_argument(
            "--embedding-key",
            type=str,
            default=SpecForgeArgs.embedding_key,
            help="The key of the embedding weight to load from the target model",
        )
        parser.add_argument(
            "--lm-head-key",
            type=str,
            default=SpecForgeArgs.lm_head_key,
            help="The key of the lm head weight to load from the target model",
        )
        parser.add_argument("--train-data-path", type=str, required=True)
        parser.add_argument("--eval-data-path", type=str, default=None)
        parser.add_argument("--output-dir", type=str, required=True)

        # add training-related arguments
        parser.add_argument(
            "--max-grad-norm", type=float, default=SpecForgeArgs.max_grad_norm
        )
        parser.add_argument(
            "--build-dataset-num-proc",
            type=int,
            default=SpecForgeArgs.build_dataset_num_proc,
        )
        parser.add_argument(
            "--dist-timeout",
            type=int,
            default=SpecForgeArgs.dist_timeout,
            help="The timeout for distributed training. If not set, will use the default value from specforge.distributed.init_distributed",
        )
        parser.add_argument(
            "--target-tp-size",
            "--tp-size",
            type=int,
            default=SpecForgeArgs.target_tp_size,
        )
        parser.add_argument(
            "--draft-tp-size", type=int, default=SpecForgeArgs.draft_tp_size
        )
        parser.add_argument(
            "--target-model-backend",
            type=str,
            default=SpecForgeArgs.target_model_backend,
            choices=["sglang", "hf", "custom"],
        )
        parser.add_argument(
            "--draft-global-batch-size",
            "--batch-size",
            type=int,
            default=SpecForgeArgs.draft_global_batch_size,
        )
        parser.add_argument(
            "--draft-micro-batch-size",
            type=int,
            default=SpecForgeArgs.draft_micro_batch_size,
        )
        parser.add_argument("--num-epochs", type=int, default=SpecForgeArgs.num_epochs)
        parser.add_argument(
            "--learning-rate", type=float, default=SpecForgeArgs.learning_rate
        )
        parser.add_argument("--max-length", type=int, default=SpecForgeArgs.max_length)
        parser.add_argument(
            "--warmup-ratio", type=float, default=SpecForgeArgs.warmup_ratio
        )
        parser.add_argument(
            "--ttt-length",
            type=int,
            default=SpecForgeArgs.ttt_length,
            help="The length for Test-Time Training (TTT).",
        )
        parser.add_argument(
            "--is-vlm", action="store_true", help="Whether the target model is a VLM"
        )
        parser.add_argument(
            "--chat-template",
            type=str,
            default=SpecForgeArgs.chat_template,
        )
        parser.add_argument(
            "--enable-zero2",
            action="store_true",
            help="enabled to shard the optimizer state; if enabled we cannot change number of GPUs for resume training",
        )
        # other args
        parser.add_argument("--cache-dir", type=str, default=SpecForgeArgs.cache_dir)
        parser.add_argument(
            "--log-interval", type=int, default=SpecForgeArgs.log_interval
        )
        parser.add_argument(
            "--eval-interval", type=int, default=SpecForgeArgs.eval_interval
        )
        parser.add_argument(
            "--save-interval", type=int, default=SpecForgeArgs.save_interval
        )
        parser.add_argument(
            "--total-steps",
            type=int,
            default=None,
            help="Total training steps. If not provided, will be calculated as num_epochs * steps_per_epoch",
        )
        parser.add_argument(
            "--max-num-saved-checkpoints",
            type=int,
            default=SpecForgeArgs.max_num_saved_checkpoints,
            help="The total number of checkpoints to save. If -1, save all checkpoints. Previous Checkpoints will be deleted.",
        )
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument(
            "--draft-attention-backend",
            type=str,
            default=SpecForgeArgs.draft_attention_backend,
            choices=["flex_attention", "sdpa"],
        )

        # resume
        parser.add_argument("--resume", action="store_true")

        # report backend
        parser.add_argument(
            "--report-to",
            type=str,
            default=SpecForgeArgs.report_to,
            choices=["wandb", "tensorboard", "swanlab", "mlflow", "none"],
            help="The integration to report results and logs to.",
        )
        # wandb-specific args
        parser.add_argument(
            "--wandb-project",
            type=str,
            default=SpecForgeArgs.wandb_project,
            help="The project name for W&B.",
        )
        parser.add_argument(
            "--wandb-name",
            type=str,
            default=SpecForgeArgs.wandb_name,
            help="The run name for W&B.",
        )
        parser.add_argument(
            "--wandb-key",
            type=str,
            default=SpecForgeArgs.wandb_key,
            help="W&B API key.",
        )
        # add swanlab-specific args ---
        parser.add_argument(
            "--swanlab-project",
            type=str,
            default=SpecForgeArgs.swanlab_project,
            help="The project name for swanlab.",
        )
        parser.add_argument(
            "--swanlab-name",
            type=str,
            default=SpecForgeArgs.swanlab_name,
            help="The experiment name for swanlab.",
        )
        parser.add_argument(
            "--swanlab-key",
            type=str,
            default=SpecForgeArgs.swanlab_key,
            help="The API key for swanlab non-interactive login.",
        )
        # mlflow-specific args
        parser.add_argument(
            "--mlflow-tracking-uri",
            type=str,
            default=SpecForgeArgs.mlflow_tracking_uri,
            help="The MLflow tracking URI. If not set, uses MLFLOW_TRACKING_URI environment variable or defaults to local './mlruns'.",
        )
        parser.add_argument(
            "--mlflow-experiment-name",
            type=str,
            default=SpecForgeArgs.mlflow_experiment_name,
            help="The MLflow experiment name. If not set, uses MLFLOW_EXPERIMENT_NAME environment variable.",
        )
        parser.add_argument(
            "--mlflow-run-name",
            type=str,
            default=SpecForgeArgs.mlflow_run_name,
            help="The MLflow run name. If not set, MLflow will auto-generate one.",
        )
        parser.add_argument(
            "--is-preformatted",
            action="store_true",
            help="Whether the input data is preformatted text with the chat template already applied to the conversation messages.",
        )

        parser.add_argument("--profile-start-step", type=int, default=30)
        parser.add_argument("--profile-num-steps", type=int, default=4)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        if dist.is_initialized():
            world_size = dist.get_world_size()
            # Parallelism Check
            if args.target_tp_size >= args.draft_tp_size:
                assert (
                    args.target_tp_size % args.draft_tp_size == 0
                ), f"target_tp_size={args.target_tp_size} must be divisible by draft_tp_size={args.draft_tp_size}"
            else:
                assert (
                    args.draft_tp_size % args.target_tp_size == 0
                ), f"draft_tp_size={args.draft_tp_size} must be divisible by target_tp_size={args.target_tp_size}"
            args.target_dp_size = world_size // args.target_tp_size
            args.draft_dp_size = world_size // args.draft_tp_size
        else:
            assert (
                args.target_tp_size == 1
            ), "not distributed training, target_tp_size must be 1"
            assert (
                args.draft_tp_size == 1
            ), "not distributed training, draft_tp_size must be 1"
            args.target_dp_size = 1
            args.draft_dp_size = 1

        # GA Check
        args.draft_accumulation_steps = (
            args.draft_global_batch_size
            // args.draft_dp_size
            // args.draft_micro_batch_size
        )
        assert (
            args.draft_accumulation_steps
            * args.draft_micro_batch_size
            * args.draft_dp_size
            == args.draft_global_batch_size
        ), f"{args.draft_global_batch_size=} must be divisible by {args.draft_dp_size=} and {args.draft_micro_batch_size=}"
        print_on_rank0(
            f"({args.draft_accumulation_steps=}) = ({args.draft_global_batch_size=}) // ({args.draft_dp_size=}) // ({args.draft_micro_batch_size=})"
        )

        args.save_per_epoch = args.save_interval == -1
        if args.resume and os.path.isdir(args.output_dir):
            args.draft_model_last_checkpoint = get_last_checkpoint(
                args.output_dir, prefix="epoch" if args.save_per_epoch else "step"
            )
            print_on_rank0(
                f"Last checkpoint detected: {args.draft_model_last_checkpoint}"
            )
        else:
            args.draft_model_last_checkpoint = None
            print_on_rank0(f"Training from scratch")

        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


def parse_specforge_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpecForge arguments")
    SpecForgeArgs.add_cli_args(parser)
    args = parser.parse_args()
    return args
