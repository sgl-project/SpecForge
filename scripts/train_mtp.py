#!/usr/bin/env python3
# coding=utf-8
"""Multi-Token Prediction (MTP) training script for Qwen3.5."""

import argparse
import functools
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
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from datasets import load_dataset
from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.core.mtp import OnlineMTPModel
from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.modeling import get_eagle3_target_model
from specforge.modeling.draft.mtp import Qwen3_5MTPDraftModel
from specforge.modeling.target.eagle3_target_model import Eagle3TargetModel
from specforge.modeling.target.mtp_target_model import generate_mtp_data
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker
from specforge.utils import (
    get_last_checkpoint,
    get_local_device,
    print_on_rank0,
    print_with_rank,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen3.5 MTP Draft Model")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--target-model-backend",
        type=str,
        default="sglang",
        choices=["sglang", "hf", "custom"],
        help="Backend for target model: 'sglang' (service), 'hf' (local), or 'custom'",
    )
    model_group.add_argument("--draft-config-path", type=str, default=None)
    model_group.add_argument(
        "--share-lm-head",
        action="store_true",
        default=True,
        help="Share lm_head weights with the target model (default: True).",
    )
    model_group.add_argument(
        "--no-share-lm-head",
        action="store_true",
        help="Disable sharing lm_head weights with the target model.",
    )
    model_group.add_argument(
        "--no-init-from-native-mtp",
        action="store_true",
        help="Do not initialize the MTP module from the native pretrained "
        "mtp.* weights in the target checkpoint (train from scratch instead). "
        "By default the native mtp.* weights are loaded for finetuning.",
    )
    model_group.add_argument(
        "--attention-backend",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention backend for the MTP draft model.",
    )
    model_group.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code"
    )
    model_group.add_argument(
        "--embedding-key",
        type=str,
        default=None,
        help="Embedding weight key in the target model. "
        "Default: 'model.embed_tokens.weight' for standard models, "
        "'model.language_model.embed_tokens.weight' for multimodal models like Qwen3.5-A3B.",
    )
    model_group.add_argument(
        "--lm-head-key",
        type=str,
        default=None,
        help="LM head weight key in the target model. Default: 'lm_head.weight'.",
    )
    model_group.add_argument(
        "--ploss-decay",
        type=float,
        default=1.0,
        help="Per-layer loss decay. For single-layer MTP this is a no-op.",
    )

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train-data-path", type=str, required=True)
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument("--chat-template", type=str, default="qwen")
    dataset_group.add_argument("--is-preformatted", action="store_true")
    dataset_group.add_argument("--dataloader-num-workers", type=int, default=8)
    dataset_group.add_argument(
        "--build-dataset-num-proc",
        type=int,
        default=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 8)),
    )

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=6)
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=6e-4)
    training_group.add_argument("--max-length", type=int, default=3072)
    training_group.add_argument("--warmup-ratio", type=float, default=0.04)
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

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="The size of the tensor parallel for the target model",
    )

    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)

    dist_group = parser.add_argument_group("distributed")
    dist_group.add_argument("--dist-timeout", type=int, default=30)

    # Only expose SGLang-specific CLI args when the backend is actually sglang.
    # This lets NPU/HF-only runs proceed without sglang installed.
    tmp_args, _ = parser.parse_known_args()
    if tmp_args.target_model_backend == "sglang":
        sglang_group = parser.add_argument_group("sglang backend")
        SGLangBackendArgs.add_args(sglang_group)

    return parser.parse_args()


def build_models(
    args,
) -> Tuple[Eagle3TargetModel, Qwen3_5MTPDraftModel]:
    """Build target model backend and MTP draft model."""
    print_on_rank0(
        f"Loading target model from {args.target_model_path} using {args.target_model_backend} backend"
    )

    target_model_kwargs = {}
    if args.target_model_backend == "sglang":
        target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()

    device = get_local_device()
    device_type = device.type

    target_model = get_eagle3_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device=device_type if args.target_model_backend != "sglang" else None,
        trust_remote_code=args.trust_remote_code,
        **target_model_kwargs,
    )

    if args.draft_config_path:
        draft_config = AutoConfig.from_pretrained(args.draft_config_path)
        print_on_rank0(f"Loaded draft config from {args.draft_config_path}")
    else:
        draft_config = AutoConfig.from_pretrained(args.target_model_path)
        print_on_rank0("Auto-generated draft config from target model")

    # MTP uses a single-layer transformer on top of the target last hidden state.
    draft_config.num_hidden_layers = 1
    draft_config._attn_implementation = args.attention_backend

    share_lm_head = args.share_lm_head and not args.no_share_lm_head
    if not hasattr(draft_config, "mtp_config") or draft_config.mtp_config is None:
        draft_config.mtp_config = {}
    draft_config.mtp_config["share_lm_head"] = share_lm_head

    draft_model = Qwen3_5MTPDraftModel(draft_config).to(
        device=device, dtype=torch.bfloat16
    )

    # Optionally initialize the MTP module from the native pretrained mtp.*
    # weights shipped with the target checkpoint (finetune) instead of random
    # init. The native keys (mtp.layers.0.*, mtp.fc.weight, mtp.norm.weight,
    # mtp.pre_fc_norm_*) match the flat draft layout; embed_tokens and lm_head
    # are skipped here and loaded separately via target sharing below.
    if not args.no_init_from_native_mtp:
        try:
            import glob as _glob

            from safetensors.torch import safe_open as _safe_open

            native_mtp: dict = {}
            for _st in sorted(
                _glob.glob(os.path.join(args.target_model_path, "*.safetensors"))
            ):
                with _safe_open(_st, framework="pt") as _f:
                    for _k in _f.keys():
                        if _k.startswith("mtp."):
                            native_mtp[_k] = _f.get_tensor(_k)
            if native_mtp:
                draft_model.load_state_dict(native_mtp, strict=False)
                print_on_rank0(
                    f"Initialized MTP module from {len(native_mtp)} native "
                    f"mtp.* weights in {args.target_model_path} (finetune)."
                )
            else:
                print_on_rank0(
                    "No native mtp.* weights found in target checkpoint; "
                    "training MTP from scratch."
                )
        except Exception as e:
            print_on_rank0(f"Could not load native mtp weights ({e}); from scratch.")

    print_on_rank0(
        f"Draft config: hidden_size={draft_config.hidden_size}, "
        f"num_hidden_layers={draft_config.num_hidden_layers}, "
        f"share_lm_head={share_lm_head}"
    )
    print_on_rank0(
        f"Draft model parameters: {sum(p.numel() for p in draft_model.parameters()):,}"
    )

    # The target model is only used to produce hidden states; keep it in eval mode
    # to disable dropout / batch-norm updates and reduce memory spikes.
    # Different backends wrap the underlying HF model differently (e.g.
    # HFEagle3TargetModel holds it in self.model), so eval the inner model when
    # the wrapper itself is not an nn.Module.
    model_for_eval = getattr(target_model, "model", target_model)
    if hasattr(model_for_eval, "eval"):
        model_for_eval.eval()

    return target_model, draft_model


def build_dataloader(args, tokenizer) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build train and eval dataloaders."""
    import hashlib

    cache_params_string = (
        f"{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    train_eagle3_dataset = build_eagle3_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        is_preformatted=args.is_preformatted,
        cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=cache_key,
        num_proc=args.build_dataset_num_proc,
    )

    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=get_dp_group(),
    )

    eval_dataloader = None
    if args.eval_data_path:
        eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
        eval_eagle3_dataset = build_eagle3_dataset(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            is_preformatted=args.is_preformatted,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle=False,
            process_group=get_dp_group(),
        )

    return train_dataloader, eval_dataloader


def save_checkpoint(args, epoch, step, mtp_model, draft_model, optimizer):
    """Save checkpoint."""
    save_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()

    with FSDP.state_dict_type(mtp_model, StateDictType.FULL_STATE_DICT):
        state_dict = mtp_model.state_dict()
        # Strip the outer wrapper prefix; keep the `mtp.*` prefix required by
        # SGLang. Keep a copy of embed_tokens so that the checkpoint is
        # self-contained for vLLM/SGLang serving, which instantiate their own
        # ``mtp.embed_tokens`` and expect it in the checkpoint. Drop the lm_head
        # only when it is shared with the target model (it is restored from the
        # target at inference via sharing); when ``--not-share-lm-head`` is set
        # the draft owns a separate lm_head and we keep it.
        if args.not_share_lm_head:
            draft_state_dict = {
                k.replace("draft_model.", ""): v
                for k, v in state_dict.items()
                if "draft_model." in k
            }
        else:
            draft_state_dict = {
                k.replace("draft_model.", ""): v
                for k, v in state_dict.items()
                if "draft_model." in k and "lm_head" not in k
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

            modeling_src = os.path.join(
                os.path.dirname(__file__),
                "..",
                "specforge",
                "modeling",
                "draft",
                "mtp.py",
            )
            modeling_dst = os.path.join(save_dir, "mtp.py")
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
        f"{mode.capitalize()} - Step {global_step} "
        f"[{global_step}/{args.num_epochs * len(train_dataloader) // args.accumulation_steps}?], "
        f"Loss: {loss:.4f}, Acc: {accuracy:.4f}"
    )

    tracker.log(logdict, step=global_step)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.getLogger().setLevel(logging.INFO)
    warnings.filterwarnings(
        "ignore",
        "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed",
    )

    args = parse_args()
    set_seed(args.seed)

    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank("Initialized distributed")

    draft_model_last_checkpoint = None
    ckpt_info = (0, 0)
    if args.resume and os.path.isdir(args.output_dir):
        draft_model_last_checkpoint, ckpt_info = get_last_checkpoint(args.output_dir)
        print(f"Last checkpoint detected: {draft_model_last_checkpoint}")

    if draft_model_last_checkpoint:
        checkpoint_config_path = os.path.join(
            draft_model_last_checkpoint, "config.json"
        )
        if os.path.exists(checkpoint_config_path):
            print(f"Loading draft config from checkpoint: {checkpoint_config_path}")
            args.draft_config_path = checkpoint_config_path

    target_model, draft_model = build_models(args)

    resume_state = None
    if draft_model_last_checkpoint:
        loaded_model = Qwen3_5MTPDraftModel.from_pretrained(
            draft_model_last_checkpoint, torch_dtype=torch.bfloat16
        )
        # strict=False: the checkpoint no longer persists embed_tokens / lm_head
        # (they are frozen/shared and restored from the target below).
        draft_model.load_state_dict(loaded_model.state_dict(), strict=False)
        del loaded_model
        print("Loaded draft model weights from checkpoint")

        training_state_path = os.path.join(
            draft_model_last_checkpoint, "training_state.pt"
        )
        if os.path.exists(training_state_path):
            resume_state = torch.load(
                training_state_path, map_location="cpu", weights_only=False
            )
            print(
                f"Will resume from epoch {resume_state['epoch']}, "
                f"step {resume_state['global_step']}"
            )

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    train_dataloader, eval_dataloader = build_dataloader(args, tokenizer)

    steps_per_epoch = math.ceil(len(train_dataloader) / args.accumulation_steps)
    total_steps = args.num_epochs * steps_per_epoch
    print_on_rank0(f"Total training steps: {total_steps}")

    print_on_rank0("Loading target embeddings and head...")
    device = get_local_device()
    device_type = device.type
    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key=args.embedding_key,
        lm_head_key=args.lm_head_key,
        device=device_type,
        trust_remote_code=args.trust_remote_code,
    )

    # Share/freeze target embeddings and lm_head with the draft model.
    draft_model.embed_tokens.weight = target_components.embed_tokens.weight
    if args.share_lm_head and not args.no_share_lm_head:
        draft_model.mtp.lm_head.weight = target_components.lm_head.weight
    draft_model.embed_tokens.requires_grad_(False)
    draft_model.mtp.lm_head.requires_grad_(False)
    print_on_rank0(
        "Shared target embed_tokens and lm_head with the MTP draft model (frozen)."
    )

    mtp_model = OnlineMTPModel(
        draft_model=draft_model,
        ploss_decay=args.ploss_decay,
    )

    fsdp_kwargs = dict(
        use_orig_params=True,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    block_names = set(getattr(draft_model, "_no_split_modules", None) or [])
    block_classes = {
        type(m) for m in mtp_model.modules() if type(m).__name__ in block_names
    }
    if block_classes:
        fsdp_kwargs["auto_wrap_policy"] = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=block_classes,
        )
    else:
        print_with_rank(
            "No _no_split_modules on draft model; falling back to single-unit "
            "FSDP wrap (no compute-comm overlap)."
        )
    mtp_model = FSDP(mtp_model, **fsdp_kwargs)
    print_with_rank("Initialized FSDP")

    start_epoch = ckpt_info[0]
    global_step = ckpt_info[1]

    optimizer = BF16Optimizer(
        draft_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_steps,
    )

    if resume_state is not None:
        optimizer.load_state_dict(resume_state)
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        del resume_state
        print_on_rank0(
            f"Restored optimizer/scheduler state: "
            f"epoch={start_epoch}, step={global_step}, "
            f"lr={optimizer.get_learning_rate():.6f}"
        )

    skip_steps = global_step - start_epoch * len(train_dataloader)

    print_on_rank0(f"Initializing tracker (report_to={args.report_to})...")
    tracker = create_tracker(args, args.output_dir)
    print_on_rank0("Tracker initialized successfully.")

    last_time = time.time()
    print_on_rank0(f"Starting training from epoch {start_epoch}, step {global_step}")

    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        draft_model.train()

        if dist.get_rank() == 0:
            progress_bar = tqdm(
                train_dataloader, desc=f"Training Epoch {epoch}", leave=True
            )
        else:
            progress_bar = train_dataloader

        for step_in_epoch, data in enumerate(progress_bar):
            if epoch == start_epoch and step_in_epoch < skip_steps:
                continue
            global_step += 1

            input_ids = data["input_ids"].to(device_type, non_blocking=True)
            attention_mask = data["attention_mask"].to(device_type, non_blocking=True)
            loss_mask = data["loss_mask"].to(device_type, non_blocking=True)

            target_output = generate_mtp_data(
                target_model, input_ids, attention_mask, loss_mask
            )
            hidden_states = target_output.last_hidden_states.to(
                device_type, non_blocking=True
            )

            loss, acc_corrects, acc_denoms = mtp_model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
                attention_mask=attention_mask,
            )

            # Aggregate per-layer metrics (single layer for Qwen3.5 MTP).
            corrects = acc_corrects[0].sum()
            denoms = acc_denoms[0].sum()
            dist.all_reduce(corrects)
            dist.all_reduce(denoms)
            accuracy = (corrects / denoms.clamp_min(1)).item()

            (loss / args.accumulation_steps).backward()

            if global_step % args.accumulation_steps == 0:
                optimizer.step()

            if global_step % args.log_interval == 0:
                loss_log = loss.clone()
                dist.all_reduce(loss_log)
                loss_log = loss_log / dist.get_world_size()

                record_metrics(
                    args,
                    loss_log.item(),
                    accuracy,
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
                        "acc": f"{accuracy:.4f}",
                        "iter_time": f"{elapsed:.2f}s",
                    }
                )

            if global_step % args.save_interval == 0:
                save_checkpoint(
                    args, epoch, global_step, mtp_model, draft_model, optimizer
                )

    save_checkpoint(
        args, args.num_epochs, global_step, mtp_model, draft_model, optimizer
    )

    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
