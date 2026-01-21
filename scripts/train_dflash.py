#!/usr/bin/env python3
# coding=utf-8
"""DFlash Training Script."""

import argparse
import logging
import math
import os
import shutil
import time
import warnings
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from datasets import load_dataset
from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.core.dflash import OnlineDFlashModel
from specforge.data import (
    build_eagle3_dataset,
    build_offline_dflash_dataset,
    prepare_dp_dataloaders,
)
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.target.dflash_target_model import get_dflash_target_model
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker
from specforge.utils import print_on_rank0, print_with_rank


def parse_args():
    parser = argparse.ArgumentParser(description="Train DFlash Draft Model")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--target-model-backend",
        type=str,
        default="hf",
        choices=["sglang", "hf"],
        help="Backend for target model: 'sglang' (service) or 'hf' (local)",
    )
    model_group.add_argument("--draft-config-path", type=str, default=None)
    model_group.add_argument("--block-size", type=int, default=16)
    model_group.add_argument("--num-draft-layers", type=int, default=1)
    model_group.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help="MASK token ID. If not provided, auto-detect from tokenizer.",
    )
    model_group.add_argument(
        "--attention-backend",
        type=str,
        default="flex_attention",
        choices=["eager", "sdpa", "flex_attention"],
        help="Attention backend for draft model.",
    )

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument(
        "--train-data-path",
        type=str,
        default=None,
        help="Path to training data (required for online mode)",
    )
    dataset_group.add_argument(
        "--train-hidden-states-path",
        type=str,
        default=None,
        help="Path to pre-computed hidden states for offline training",
    )
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument(
        "--eval-hidden-states-path",
        type=str,
        default=None,
        help="Path to pre-computed hidden states for offline evaluation",
    )
    dataset_group.add_argument("--chat-template", type=str, default="qwen")
    dataset_group.add_argument("--is-preformatted", action="store_true")
    dataset_group.add_argument("--dataloader-num-workers", type=int, default=8)
    dataset_group.add_argument(
        "--build-dataset-num-proc",
        type=int,
        default=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 8)),
    )

    model_group.add_argument(
        "--lm-head-key",
        type=str,
        default="lm_head.weight",
        help="The key of the lm head weight to load from the target model (for offline training)",
    )

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=3)
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=1e-4)
    training_group.add_argument("--max-length", type=int, default=2048)
    training_group.add_argument("--warmup-ratio", type=float, default=0.01)
    training_group.add_argument("--max-grad-norm", type=float, default=1.0)
    training_group.add_argument("--accumulation-steps", type=int, default=1)
    training_group.add_argument("--seed", type=int, default=42)
    training_group.add_argument("--resume", action="store_true")

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output-dir", type=str, required=True)
    output_group.add_argument("--cache-dir", type=str, default="./cache")
    output_group.add_argument("--log-interval", type=int, default=50)
    output_group.add_argument("--eval-interval", type=int, default=1000)
    output_group.add_argument("--save-interval", type=int, default=1000)

    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)

    dist_group = parser.add_argument_group("distributed")
    dist_group.add_argument("--dist-timeout", type=int, default=30)

    # SGLang specific args
    sglang_group = parser.add_argument_group("sglang backend")
    SGLangBackendArgs.add_args(sglang_group)

    return parser.parse_args()


def build_target_model(args, draft_model: DFlashDraftModel, is_online: bool = True):
    """Build target model based on training mode.

    For online training: Build full target model wrapper.
    For offline training: Build only TargetHead (lm_head weights).
    """
    if is_online:
        print_on_rank0(
            f"Loading target model from {args.target_model_path} using {args.target_model_backend} backend"
        )
        target_model_kwargs = {}
        if args.target_model_backend == "sglang":
            target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()

        target_model = get_dflash_target_model(
            pretrained_model_name_or_path=args.target_model_path,
            backend=args.target_model_backend,
            torch_dtype=torch.bfloat16,
            device="cuda" if args.target_model_backend == "hf" else None,
            **target_model_kwargs,
        )
        # Set capture layers for target model based on draft model config
        target_model.set_capture_layers(draft_model.target_layer_ids)
        return target_model
    else:
        # For offline training, we don't need the full target model
        # Hidden states are already pre-computed
        print_on_rank0(
            "Offline mode: Skipping target model loading (using pre-computed hidden states)"
        )
        return None


def build_draft_model(args) -> DFlashDraftModel:
    """Build draft model."""
    if args.draft_config_path:
        draft_config = AutoConfig.from_pretrained(args.draft_config_path)
        print_on_rank0(f"Loaded draft config from {args.draft_config_path}")
    else:
        # Load config from HF (needed for structure info even if backend is sglang)
        target_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config.num_hidden_layers = args.num_draft_layers
        draft_config.block_size = args.block_size
        draft_config.num_target_layers = target_config.num_hidden_layers
        print_on_rank0("Auto-generated draft config from target model")

    # Set attention implementation based on backend
    draft_config._attn_implementation = args.attention_backend
    print_on_rank0(f"Using attention backend: {args.attention_backend}")

    draft_model = DFlashDraftModel(draft_config).cuda().to(torch.bfloat16)

    print_on_rank0(
        f"Draft config: block_size={draft_config.block_size}, "
        f"num_hidden_layers={draft_config.num_hidden_layers}, "
        f"num_target_layers={draft_config.num_target_layers}"
    )
    print_on_rank0(
        f"Draft model parameters: {sum(p.numel() for p in draft_model.parameters()):,}"
    )

    return draft_model


def build_dataloader(
    args, tokenizer, is_online: bool = True
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build train and eval dataloaders.

    For online training: Build from conversation data.
    For offline training: Build from pre-computed hidden states.
    """
    import hashlib

    # Common filtering threshold: DFlash requires >= 2 * block_size loss tokens
    min_loss_tokens = 2 * args.block_size

    if is_online:
        # Online mode: Build from conversation data
        cache_params_string = (
            f"{args.train_data_path}-"
            f"{args.max_length}-"
            f"{args.chat_template}-"
            f"{args.target_model_path}"
        )
        cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

        train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
        train_dflash_dataset = build_eagle3_dataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            is_preformatted=args.is_preformatted,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=cache_key,
            num_proc=args.build_dataset_num_proc,
        )

        # Filter out samples with too few loss tokens
        original_size = len(train_dflash_dataset)
        train_dflash_dataset = train_dflash_dataset.filter(
            lambda x: x["loss_mask"].sum() >= min_loss_tokens
        )
        print_on_rank0(
            f"Filtered train dataset: {original_size} -> {len(train_dflash_dataset)} samples"
        )
    else:
        # Offline mode: Build from pre-computed hidden states
        # Note: Filtering is already done in prepare_hidden_states.py, no need to filter here
        print_on_rank0(
            f"Loading offline train dataset from {args.train_hidden_states_path}"
        )
        train_dflash_dataset = build_offline_dflash_dataset(
            hidden_states_path=args.train_hidden_states_path,
            max_len=args.max_length,
            block_size=args.block_size,
        )

    train_dataloader = prepare_dp_dataloaders(
        train_dflash_dataset,
        args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=get_dp_group(),
        requires_target=False,
    )

    eval_dataloader = None
    if is_online and args.eval_data_path:
        eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
        eval_dflash_dataset = build_eagle3_dataset(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            is_preformatted=args.is_preformatted,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_dflash_dataset,
            args.batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle=False,
            process_group=get_dp_group(),
            requires_target=False,
        )
    elif not is_online and args.eval_hidden_states_path:
        print_on_rank0(
            f"Loading offline eval dataset from {args.eval_hidden_states_path}"
        )
        eval_dflash_dataset = build_offline_dflash_dataset(
            hidden_states_path=args.eval_hidden_states_path,
            max_len=args.max_length,
            block_size=args.block_size,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_dflash_dataset,
            args.batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle=False,
            process_group=get_dp_group(),
            requires_target=False,
        )

    return train_dataloader, eval_dataloader


def save_checkpoint(args, epoch, step, dflash_model, draft_model, optimizer):
    """Save checkpoint."""
    save_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()

    with FSDP.state_dict_type(dflash_model, StateDictType.FULL_STATE_DICT):
        state_dict = dflash_model.state_dict()
        draft_state_dict = {
            k.replace("draft_model.", ""): v
            for k, v in state_dict.items()
            if "draft_model." in k
        }

        if dist.get_rank() == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": step,
                    "args": args,
                    **optimizer.state_dict(),
                },
                os.path.join(save_dir, "training_state.pt"),
            )

            draft_model.save_pretrained(save_dir, state_dict=draft_state_dict)

            # Copy modeling_dflash.py for inference compatibility
            modeling_src = os.path.join(
                os.path.dirname(__file__),
                "..",
                "specforge",
                "modeling",
                "draft",
                "dflash.py",
            )
            modeling_dst = os.path.join(save_dir, "modeling_dflash.py")
            if os.path.exists(modeling_src):
                shutil.copy(modeling_src, modeling_dst)

            print_on_rank0(f"Saved checkpoint to {save_dir}")

    dist.barrier()


def record_metrics(
    args,
    loss: float,
    accuracy: float,
    global_step: int,
    tracker,
    optimizer,
    train_dataloader=None,
    mode: str = "train",
) -> None:
    logdict = {}

    if mode == "train" and optimizer is not None:
        logdict["train/lr"] = optimizer.get_learning_rate()

    logdict[f"{mode}/loss"] = loss
    logdict[f"{mode}/accuracy"] = accuracy

    print_on_rank0(
        f"{mode.capitalize()} - Step {global_step} [{global_step}/{args.num_epochs * len(train_dataloader) // args.accumulation_steps}?], Loss: {loss:.4f}, Acc: {accuracy:.4f}"
    )

    tracker.log(logdict, step=global_step)


def main():
    # Configure logging to ensure we see INFO logs
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Force the root logger to INFO as well, just in case
    logging.getLogger().setLevel(logging.INFO)

    # Filter annoying FSDP warnings
    warnings.filterwarnings(
        "ignore",
        "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed",
    )

    args = parse_args()
    set_seed(args.seed)

    init_distributed(timeout=args.dist_timeout)
    print_with_rank("Initialized distributed")

    # Determine training mode and validate required arguments
    is_online = args.train_hidden_states_path is None
    print_on_rank0(f"Training mode: {'online' if is_online else 'offline'}")

    if is_online:
        if args.train_data_path is None:
            raise ValueError("--train-data-path is required for online training mode")
    else:
        if not os.path.exists(args.train_hidden_states_path):
            raise ValueError(
                f"Hidden states path not found: {args.train_hidden_states_path}"
            )

    # Build draft model first (needed for target model layer config)
    draft_model = build_draft_model(args)

    # Build target model (None for offline mode)
    target_model = build_target_model(args, draft_model, is_online)

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    # Get mask_token_id
    if args.mask_token_id is not None:
        mask_token_id = args.mask_token_id
    elif tokenizer.mask_token_id is not None:
        mask_token_id = tokenizer.mask_token_id
    else:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        mask_token_id = tokenizer.mask_token_id
    print_on_rank0(f"Using mask_token_id: {mask_token_id}")

    train_dataloader, eval_dataloader = build_dataloader(args, tokenizer, is_online)

    steps_per_epoch = math.ceil(len(train_dataloader) / args.accumulation_steps)
    total_steps = args.num_epochs * steps_per_epoch
    print_on_rank0(f"Total training steps: {total_steps}")

    # Note: We need embedding layer for DFlash wrapper.
    # For SGLang backend, we can't easily get the embedding layer object.
    # We use TargetEmbeddingsAndHead to efficiently load only needed weights.
    print_on_rank0("Loading target embeddings and head efficiently...")
    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key="model.embed_tokens.weight",  # Adjust if Qwen/Llama differs
        lm_head_key=args.lm_head_key,
        device="cuda",
    )

    dflash_model = OnlineDFlashModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=draft_model.block_size,
        mask_token_id=mask_token_id,
        attention_backend=args.attention_backend,
    )

    dflash_model = FSDP(
        dflash_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    print_with_rank("Initialized FSDP")

    optimizer = BF16Optimizer(
        draft_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_steps,
    )

    print_on_rank0(f"Initializing tracker (report_to={args.report_to})...")
    tracker = create_tracker(args, args.output_dir)
    print_on_rank0("Tracker initialized successfully.")

    global_step = 0
    last_time = time.time()

    for epoch in range(args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        draft_model.train()

        if dist.get_rank() == 0:
            progress_bar = tqdm(
                train_dataloader, desc=f"Training Epoch {epoch}", leave=True
            )
        else:
            progress_bar = train_dataloader

        for data in progress_bar:
            global_step += 1

            input_ids = data["input_ids"].cuda()
            attention_mask = data["attention_mask"].cuda()
            loss_mask = data["loss_mask"].cuda()

            if is_online:
                # Online mode: Generate hidden states from target model
                target_output = target_model.generate_dflash_data(
                    input_ids, attention_mask, loss_mask
                )
                hidden_states = target_output.hidden_states.cuda()
            else:
                # Offline mode: Load pre-computed hidden states from dataset
                hidden_states = data["hidden_state"].cuda()

            # Forward pass (Parallel Training)
            loss, accuracy = dflash_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
            )

            (loss / args.accumulation_steps).backward()

            if global_step % args.accumulation_steps == 0:
                optimizer.step()

            if global_step % args.log_interval == 0:
                loss_log = loss.clone()
                acc_log = accuracy.clone()
                dist.all_reduce(loss_log)
                dist.all_reduce(acc_log)
                loss_log = loss_log / dist.get_world_size()
                acc_log = acc_log / dist.get_world_size()

                record_metrics(
                    args,
                    loss_log.item(),
                    acc_log.item(),
                    global_step,
                    tracker,
                    optimizer,
                    train_dataloader,
                    mode="train",
                )

            if dist.get_rank() == 0:
                elapsed = time.time() - last_time
                last_time = time.time()
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{accuracy.item():.4f}",
                        "iter_time": f"{elapsed:.2f}s",
                    }
                )

            if global_step % args.save_interval == 0:
                save_checkpoint(
                    args, epoch, global_step, dflash_model, draft_model, optimizer
                )

    save_checkpoint(
        args, args.num_epochs, global_step, dflash_model, draft_model, optimizer
    )

    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
