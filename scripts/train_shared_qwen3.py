#!/usr/bin/env python
"""Training script for single-model speculative decoding with shared backend.

This script trains a draft model that shares a frozen pretrained backbone
(Qwen3) while learning a small set of trainable draft layers with
bi-directional attention conditioned on the target model's key-value pairs.
"""

import argparse
import hashlib
import math
import os
import time
from argparse import ArgumentParser, Namespace
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import set_seed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config

from datasets import Dataset
from specforge.core.shared_backend import OnlineSharedBackendModel
from specforge.distributed import (
    destroy_distributed,
    get_dp_group,
    init_distributed,
)
from specforge.modeling.auto import AutoQwen3SharedDraftModel
from specforge.modeling.draft.qwen3_shared import Qwen3SharedDraftModel
from specforge.optimizer import BF16Optimizer
from specforge.tracker import Tracker, create_tracker, get_tracker_class
from specforge.utils import (
    get_last_checkpoint,
    print_args_with_dots,
    print_on_rank0,
    print_with_rank,
    safe_conversations_generator,
)


def parse_args() -> tuple[ArgumentParser, Namespace]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train single-model speculative decoding with shared backend"
    )

    # Model arguments
    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--target-model-path",
        type=str,
        required=True,
        help="Path to the pretrained Qwen3 model",
    )
    model_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading model",
    )
    model_group.add_argument(
        "--num-draft-layers",
        type=int,
        default=5,
        help="Number of trainable draft layers",
    )
    model_group.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Block size for parallel training",
    )
    model_group.add_argument(
        "--ce-weight",
        type=float,
        default=1.0,
        help="Weight for cross-entropy loss",
    )
    model_group.add_argument(
        "--mse-weight",
        type=float,
        default=0.1,
        help="Weight for MSE loss between K/V pairs",
    )

    # Dataset arguments
    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument(
        "--train-data-path",
        type=str,
        required=True,
        help="Path to training data",
    )
    dataset_group.add_argument(
        "--eval-data-path",
        type=str,
        default=None,
        help="Path to evaluation data",
    )
    dataset_group.add_argument(
        "--chat-template",
        type=str,
        default="qwen3",
        help="Chat template to use",
    )
    dataset_group.add_argument(
        "--is-preformatted",
        action="store_true",
        help="Whether data is preformatted",
    )
    dataset_group.add_argument(
        "--build-dataset-num-proc",
        type=int,
        default=8,
        help="Number of processes for dataset building",
    )
    dataset_group.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )

    # Training arguments
    training_group = parser.add_argument_group("training")
    training_group.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    training_group.add_argument(
        "--max-num-steps",
        type=int,
        default=None,
        help="Maximum number of training steps",
    )
    training_group.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device batch size",
    )
    training_group.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    training_group.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    training_group.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.015,
        help="Warmup ratio",
    )
    training_group.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="Maximum gradient norm for clipping",
    )
    training_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    training_group.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Checkpoint directory to resume from",
    )
    training_group.add_argument(
        "--eval-interval",
        type=int,
        default=5000,
        help="Evaluation interval in steps",
    )
    training_group.add_argument(
        "--save-interval",
        type=int,
        default=5000,
        help="Save interval in steps",
    )
    training_group.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Logging interval in steps",
    )
    training_group.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )

    # Optimization arguments
    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallelism size (not supported for shared backend)",
    )

    # Other arguments
    other_group = parser.add_argument_group("others")
    other_group.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Cache directory",
    )
    other_group.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory",
    )
    other_group.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    other_group.add_argument(
        "--dist-timeout",
        type=int,
        default=20,
        help="Distributed timeout in minutes",
    )
    other_group.add_argument(
        "--model-download-dir",
        type=str,
        default=None,
        help="Model download directory",
    )

    # Tracker arguments
    tracker_group = parser.add_argument_group("tracker")
    tracker_group.add_argument(
        "--report-to",
        type=str,
        default="none",
        choices=["wandb", "tensorboard", "swanlab", "mlflow", "none"],
        help="Tracker to use",
    )
    tracker_group.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name",
    )
    tracker_group.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="W&B run name",
    )
    tracker_group.add_argument(
        "--wandb-key",
        type=str,
        default=None,
        help="W&B API key",
    )

    args = parser.parse_args()
    return parser, args


def build_tracker(args: Namespace, parser: ArgumentParser) -> Tracker:
    """Build the experiment tracker."""
    tracker_class = get_tracker_class(args.report_to)
    if tracker_class:
        tracker_class.validate_args(parser, args)
    else:
        parser.error(f"Unknown tracker: {args.report_to}")
    tracker = create_tracker(args, args.output_dir)
    return tracker


def build_shared_model(args: Namespace) -> OnlineSharedBackendModel:
    """Build the shared backend model.

    Args:
        args: Command line arguments

    Returns:
        OnlineSharedBackendModel wrapping the draft model with target model
    """
    print_on_rank0("Loading target model...")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.model_download_dir,
    )

    print_on_rank0(f"Target model loaded: {type(target_model).__name__}")

    # Create draft model config
    draft_config = Qwen3Config.from_pretrained(
        args.target_model_path,
        trust_remote_code=args.trust_remote_code,
    )

    print_on_rank0("Creating draft model...")
    draft_model = Qwen3SharedDraftModel(
        draft_config,
        num_draft_layers=args.num_draft_layers,
        block_size=args.block_size,
    )

    # Set the target model (this will also freeze it)
    draft_model.set_target_model(target_model)
    draft_model = draft_model.to(torch.bfloat16)

    print_on_rank0(f"Draft model created with {args.num_draft_layers} layers")
    print_on_rank0(f"Target layer IDs: {draft_model.get_target_layer_ids()}")

    # Create training wrapper
    model = OnlineSharedBackendModel(
        model=draft_model,
        block_size=args.block_size,
        ce_weight=args.ce_weight,
        mse_weight=args.mse_weight,
    )

    return model


def build_dataloaders(
    args: Namespace,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """Build training and evaluation dataloaders.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (train_dataloader, eval_dataloader)
    """
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path,
        trust_remote_code=args.trust_remote_code,
    )

    # Cache parameters
    cache_params_string = (
        f"{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

    # Build training dataset
    print_on_rank0("Building training dataset...")
    train_dataset = Dataset.from_generator(
        generator=safe_conversations_generator,
        gen_kwargs={"file_path": args.train_data_path},
    )

    # Simple preprocessing: tokenize and create loss mask
    def preprocess_function(examples):
        # Tokenize
        tokenized = tokenizer(
            examples["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        # Create loss mask (1 for real tokens, 0 for padding)
        loss_mask = [
            [1] * len(input_ids) for input_ids in tokenized["input_ids"]
        ]
        tokenized["loss_mask"] = loss_mask
        return tokenized

    # For preformatted data
    if args.is_preformatted:
        train_dataset = train_dataset.map(
            lambda x: {
                "input_ids": x["input_ids"][:args.max_length],
                "attention_mask": x["attention_mask"][:args.max_length],
                "loss_mask": x.get("loss_mask", [1] * len(x["input_ids"]))[:args.max_length],
            },
            remove_columns=train_dataset.column_names,
            num_proc=args.build_dataset_num_proc,
        )
    else:
        # Apply chat template
        def apply_chat_template(example):
            if "conversations" in example:
                # Apply chat template
                text = tokenizer.apply_chat_template(
                    example["conversations"],
                    tokenize=False,
                    chat_template=args.chat_template,
                )
            else:
                text = example.get("text", "")
            return {"text": text}

        train_dataset = train_dataset.map(
            apply_chat_template,
            num_proc=args.build_dataset_num_proc,
        )

        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            num_proc=args.build_dataset_num_proc,
        )

    # Calculate global batch size
    global_batch_size = args.batch_size * dist.get_world_size()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    print_on_rank0(f"Training dataset size: {len(train_dataset)}")

    # Build eval dataloader if provided
    eval_dataloader = None
    if args.eval_data_path is not None:
        print_on_rank0("Building evaluation dataset...")
        eval_dataset = Dataset.from_generator(
            generator=safe_conversations_generator,
            gen_kwargs={"file_path": args.eval_data_path},
        )

        if not args.is_preformatted:
            eval_dataset = eval_dataset.map(
                apply_chat_template,
                num_proc=args.build_dataset_num_proc,
            )

            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=eval_dataset.column_names,
                num_proc=args.build_dataset_num_proc,
            )
        else:
            eval_dataset = eval_dataset.map(
                lambda x: {
                    "input_ids": x["input_ids"][:args.max_length],
                    "attention_mask": x["attention_mask"][:args.max_length],
                    "loss_mask": x.get("loss_mask", [1] * len(x["input_ids"]))[:args.max_length],
                },
                remove_columns=eval_dataset.column_names,
                num_proc=args.build_dataset_num_proc,
            )

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.dataloader_num_workers,
            drop_last=True,
        )
        print_on_rank0(f"Evaluation dataset size: {len(eval_dataset)}")

    return train_dataloader, eval_dataloader


def save_checkpoints(
    args: Namespace,
    epoch: int,
    step: int,
    model: nn.Module,
    optimizer: Optimizer,
):
    """Save model checkpoint."""
    epoch_output_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(epoch_output_dir, exist_ok=True)
    dist.barrier()

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        model_state_dict = model.state_dict()
        state_to_save = {
            "epoch": epoch,
            "global_step": step,
            "args": args,
        }
        state_to_save.update(optimizer.state_dict())

        # Extract only draft model parameters
        draft_model_state_dict = {
            k.replace("model.", ""): v
            for k, v in model_state_dict.items()
            if "model." in k
        }

        if dist.get_rank() == 0:
            torch.save(
                state_to_save,
                os.path.join(epoch_output_dir, "training_state.pt"),
            )
            print_on_rank0(
                f"Saved training state to {epoch_output_dir}/training_state.pt"
            )
            # Save draft model
            model.model.save_pretrained(
                epoch_output_dir,
                state_dict=draft_model_state_dict,
            )
            print_on_rank0(f"Saved model to {epoch_output_dir}")
        dist.barrier()


def run_forward(
    model: nn.Module,
    data: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run forward pass.

    Args:
        model: The training model
        data: Batch data

    Returns:
        Tuple of (loss, ce_loss, mse_loss)
    """
    outputs = model(
        input_ids=data["input_ids"].cuda(),
        attention_mask=data["attention_mask"].cuda(),
        loss_mask=data["loss_mask"].cuda(),
    )
    return outputs["loss"], outputs["ce_loss"], outputs["mse_loss"]


def run_backward_and_update(
    loss: torch.Tensor,
    optimizer: Optimizer,
) -> None:
    """Run backward pass and optimizer step."""
    loss.backward()
    optimizer.step()


def record_metrics(
    args: Namespace,
    loss: torch.Tensor,
    ce_loss: torch.Tensor,
    mse_loss: torch.Tensor,
    global_step: int,
    tracker: Tracker,
    optimizer: Optional[Optimizer] = None,
    mode: str = "train",
) -> None:
    """Record training/evaluation metrics."""
    logdict = {}

    if mode == "train" and optimizer is not None:
        logdict["train/lr"] = optimizer.get_learning_rate()

    # Average losses across all processes
    loss_tensor = torch.tensor(loss.item(), device="cuda")
    ce_loss_tensor = torch.tensor(ce_loss.item(), device="cuda")
    mse_loss_tensor = torch.tensor(mse_loss.item(), device="cuda")

    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    dist.all_reduce(ce_loss_tensor, op=dist.ReduceOp.AVG)
    dist.all_reduce(mse_loss_tensor, op=dist.ReduceOp.AVG)

    logdict[f"{mode}/loss"] = loss_tensor.item()
    logdict[f"{mode}/ce_loss"] = ce_loss_tensor.item()
    logdict[f"{mode}/mse_loss"] = mse_loss_tensor.item()

    print_on_rank0(
        f"{mode.capitalize()} - Step {global_step}: "
        f"Loss={loss_tensor.item():.4f}, "
        f"CE={ce_loss_tensor.item():.4f}, "
        f"MSE={mse_loss_tensor.item():.4f}"
    )

    tracker.log(logdict, step=global_step)


def main():
    # ================================================
    # 1. Initialize
    # ================================================
    parser, args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout)

    print_on_rank0("Starting shared backend training")
    print_args_with_dots(args)

    # ================================================
    # 2. Build model
    # ================================================
    model = build_shared_model(args)
    model = model.cuda()

    # Wrap with FSDP - only shard trainable parameters (draft model)
    # The target model is frozen and not sharded
    model = FSDP(
        model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    print_on_rank0("Initialized FSDP model")

    # Verify frozen parameters
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print_on_rank0(
        f"Total parameters: {total_params:,}, "
        f"Trainable: {trainable_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    # ================================================
    # 3. Build dataloaders
    # ================================================
    train_dataloader, eval_dataloader = build_dataloaders(args)

    # Calculate total steps
    steps_per_epoch = len(train_dataloader)
    total_steps = args.num_epochs * steps_per_epoch
    if args.max_num_steps is not None:
        total_steps = min(total_steps, args.max_num_steps)
    print_on_rank0(f"Total training steps: {total_steps}")

    # ================================================
    # 4. Build optimizer
    # ================================================
    # Only optimize trainable parameters
    optimizer = BF16Optimizer(
        model.model,  # Pass the draft model directly
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_steps,
    )
    print_on_rank0("Initialized optimizer")

    # ================================================
    # 5. Build tracker
    # ================================================
    tracker = build_tracker(args, parser)
    global_step = 0
    start_epoch = 0
    dist.barrier()

    last_time = time.time()

    # ================================================
    # 6. Training loop
    # ================================================
    print_on_rank0(f"Starting training from epoch {start_epoch}")

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        train_dataloader.sampler.set_epoch(epoch + 1)

        if dist.get_rank() == 0:
            progress_bar = tqdm(
                train_dataloader,
                desc=f"Training Epoch {epoch}",
                leave=True,
            )
        else:
            progress_bar = train_dataloader

        for data in progress_bar:
            global_step += 1

            # Forward pass
            loss, ce_loss, mse_loss = run_forward(model, data)

            # Backward pass
            optimizer.zero_grad()
            run_backward_and_update(loss, optimizer)

            # Log metrics
            if global_step % args.log_interval == 0:
                record_metrics(
                    args,
                    loss,
                    ce_loss,
                    mse_loss,
                    global_step,
                    tracker,
                    optimizer,
                    mode="train",
                )

            if dist.get_rank() == 0:
                time_per_step = time.time() - last_time
                last_time = time.time()
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "time": f"{time_per_step:.2f}s",
                    }
                )

            # Evaluation
            if (
                eval_dataloader is not None
                and global_step % args.eval_interval == 0
            ):
                model.eval()
                eval_losses = []
                eval_ce_losses = []
                eval_mse_losses = []

                for eval_data in tqdm(eval_dataloader, desc="Evaluating"):
                    with torch.no_grad():
                        eval_loss, eval_ce, eval_mse = run_forward(model, eval_data)
                        eval_losses.append(eval_loss.item())
                        eval_ce_losses.append(eval_ce.item())
                        eval_mse_losses.append(eval_mse.item())

                # Average eval losses
                avg_loss = sum(eval_losses) / len(eval_losses)
                avg_ce = sum(eval_ce_losses) / len(eval_ce_losses)
                avg_mse = sum(eval_mse_losses) / len(eval_mse_losses)

                record_metrics(
                    args,
                    torch.tensor(avg_loss, device="cuda"),
                    torch.tensor(avg_ce, device="cuda"),
                    torch.tensor(avg_mse, device="cuda"),
                    global_step,
                    tracker,
                    mode="eval",
                )

                model.train()

            # Save checkpoint
            if global_step % args.save_interval == 0:
                save_checkpoints(args, epoch, global_step, model, optimizer)

            if args.max_num_steps is not None and global_step >= args.max_num_steps:
                break

        if args.max_num_steps is not None and global_step >= args.max_num_steps:
            break

    # Save final checkpoint
    if global_step % args.save_interval != 0:
        print_on_rank0(
            f"Training completed at step {global_step}, saving final checkpoint..."
        )
        save_checkpoints(args, epoch, global_step, model, optimizer)

    # Close tracker and cleanup
    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
