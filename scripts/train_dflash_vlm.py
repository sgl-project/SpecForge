#!/usr/bin/env python3
# coding=utf-8
"""DFlash online training for Qwen-style vision-language target models."""

import argparse
import functools
import hashlib
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
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from datasets import load_dataset
from specforge.core.dflash import OnlineDFlashModel
from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.target.dflash_target_model import DFlashTargetOutput
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker
from specforge.utils import get_last_checkpoint, print_on_rank0, print_with_rank


def parse_dtype(dtype: str) -> torch.dtype:
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype}") from exc


def current_device(device_type: str) -> torch.device:
    if device_type == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    if device_type == "npu":
        return torch.device("npu", torch.npu.current_device())
    raise ValueError(f"Unsupported device_type: {device_type}")


def load_vlm_model(args, dtype: torch.dtype) -> torch.nn.Module:
    auto_model_classes = []
    try:
        from transformers import AutoModelForImageTextToText

        auto_model_classes.append(AutoModelForImageTextToText)
    except ImportError:
        pass
    try:
        from transformers import AutoModelForVision2Seq

        auto_model_classes.append(AutoModelForVision2Seq)
    except ImportError:
        pass
    auto_model_classes.append(AutoModelForCausalLM)

    last_error = None
    for model_cls in auto_model_classes:
        try:
            model = model_cls.from_pretrained(
                args.target_model_path,
                torch_dtype=dtype,
                cache_dir=args.model_download_dir,
                output_hidden_states=True,
                trust_remote_code=args.trust_remote_code,
            )
            model.eval()
            model.requires_grad_(False)
            return model.to(current_device(args.device_type))
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"Failed to load VLM target model from {args.target_model_path}"
    ) from last_error


class HFDFlashVLMTargetModel:
    """HF VLM target wrapper that returns DFlash context hidden states."""

    def __init__(
        self,
        model: torch.nn.Module,
        position_ids_mode: str = "auto",
    ):
        self.model = model
        self.capture_layer_ids = None
        self.position_ids_mode = position_ids_mode

    def set_capture_layers(self, layer_ids):
        self.capture_layer_ids = layer_ids

    def _get_rope_indexer(self):
        for owner in (self.model, getattr(self.model, "model", None)):
            if owner is not None and hasattr(owner, "get_rope_index"):
                return owner.get_rope_index
        return None

    def _compute_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if position_ids is not None:
            return position_ids
        if self.position_ids_mode == "none" or image_grid_thw is None:
            return None

        rope_indexer = self._get_rope_indexer()
        if rope_indexer is None:
            return None

        try:
            result = rope_indexer(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                second_per_grid_ts=None,
                attention_mask=attention_mask,
            )
        except TypeError:
            try:
                result = rope_indexer(
                    input_ids,
                    image_grid_thw,
                    None,
                    second_per_grid_ts=None,
                    attention_mask=attention_mask,
                )
            except TypeError:
                result = rope_indexer(input_ids, image_grid_thw, None)

        return result[0] if isinstance(result, tuple) else result

    @torch.no_grad()
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> DFlashTargetOutput:
        position_ids = self._compute_position_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
        )

        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "output_hidden_states": True,
            "use_cache": False,
        }
        if position_ids is not None:
            model_kwargs["position_ids"] = position_ids

        outputs = self.model(**model_kwargs)

        offset = 1
        if self.capture_layer_ids is None:
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = torch.cat(
                [outputs.hidden_states[idx + offset] for idx in self.capture_layer_ids],
                dim=-1,
            )

        return DFlashTargetOutput(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train DFlash Draft Model for VLMs")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--target-model-backend",
        type=str,
        default="hf",
        choices=["hf"],
        help="VLM DFlash currently uses local HF target forward.",
    )
    model_group.add_argument(
        "--draft-config-path", "--draft-model-config", dest="draft_config_path", type=str, required=True
    )
    model_group.add_argument("--block-size", type=int, default=16)
    model_group.add_argument("--mask-token-id", type=int, default=None)
    model_group.add_argument(
        "--attention-backend",
        type=str,
        default="flex_attention",
        choices=["eager", "sdpa", "flex_attention"],
    )
    model_group.add_argument("--trust-remote-code", action="store_true")
    model_group.add_argument("--num-anchors", type=int, default=512)
    model_group.add_argument("--loss-decay-gamma", type=float, default=None)
    model_group.add_argument(
        "--loss-type",
        type=str,
        default="dflash",
        choices=[
            "dflash",
            "dpace",
            "dpace-cumulative-confidence-only",
            "dpace-continuation-value-only",
        ],
    )
    model_group.add_argument("--dpace-alpha", type=float, default=0.5)
    model_group.add_argument("--embedding-key", type=str, default=None)
    model_group.add_argument("--lm-head-key", type=str, default=None)
    model_group.add_argument(
        "--position-ids-mode",
        type=str,
        default="auto",
        choices=["auto", "none"],
        help="auto computes target VLM mRoPE position_ids when the model exposes get_rope_index.",
    )

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train-data-path", type=str, required=True)
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument("--image-root", type=str, default=None)
    dataset_group.add_argument("--chat-template", type=str, default="qwen2-vl")
    dataset_group.add_argument("--is-preformatted", action="store_true")
    dataset_group.add_argument("--train-only-last-turn", action="store_true")
    dataset_group.add_argument("--dataloader-num-workers", type=int, default=4)
    dataset_group.add_argument(
        "--build-dataset-num-proc",
        type=int,
        default=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 8)),
    )
    dataset_group.add_argument("--min-pixels", type=int, default=50176)
    dataset_group.add_argument("--max-pixels", type=int, default=802816)

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=10)
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=6e-4)
    training_group.add_argument("--max-length", type=int, default=3072)
    training_group.add_argument("--warmup-ratio", type=float, default=0.04)
    training_group.add_argument("--max-grad-norm", type=float, default=1.0)
    training_group.add_argument("--accumulation-steps", type=int, default=1)
    training_group.add_argument("--seed", type=int, default=42)
    training_group.add_argument("--resume", action="store_true")
    training_group.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output-dir", type=str, required=True)
    output_group.add_argument("--cache-dir", type=str, default="./cache")
    output_group.add_argument("--model-download-dir", type=str, default=None)
    output_group.add_argument("--log-interval", type=int, default=50)
    output_group.add_argument("--eval-interval", type=int, default=1000)
    output_group.add_argument("--save-interval", type=int, default=1000)

    dist_group = parser.add_argument_group("distributed")
    dist_group.add_argument("--tp-size", type=int, default=1)
    dist_group.add_argument("--dist-timeout", type=int, default=30)
    dist_group.add_argument(
        "--device-type", type=str, default="cuda", choices=["cuda", "npu"]
    )
    dist_group.add_argument(
        "--dist-backend",
        type=str,
        default=None,
        help="Defaults to nccl for cuda and hccl for npu.",
    )

    tracker_group = parser.add_argument_group("tracker")
    from specforge.args import TrackerArgs

    TrackerArgs.add_args(tracker_group)

    return parser.parse_args()


def build_processor_and_tokenizer(args):
    processor = AutoProcessor.from_pretrained(
        args.target_model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        cache_dir=args.model_download_dir,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.target_model_path,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.model_download_dir,
        )
    return processor, tokenizer


def build_models(args, dtype: torch.dtype) -> Tuple[HFDFlashVLMTargetModel, DFlashDraftModel]:
    print_on_rank0(f"Loading VLM target model from {args.target_model_path}")
    target_model = HFDFlashVLMTargetModel(
        load_vlm_model(args, dtype=dtype),
        position_ids_mode=args.position_ids_mode,
    )

    draft_config = AutoConfig.from_pretrained(
        args.draft_config_path,
        trust_remote_code=args.trust_remote_code,
    )
    if hasattr(draft_config, "block_size") and draft_config.block_size != args.block_size:
        print_on_rank0(
            f"Warning: config block_size ({draft_config.block_size}) differs from "
            f"command-line block_size ({args.block_size}); using config value."
        )

    if not hasattr(draft_config, "dflash_config") or draft_config.dflash_config is None:
        draft_config.dflash_config = {}
    draft_config._attn_implementation = args.attention_backend

    draft_model = DFlashDraftModel(draft_config).to(current_device(args.device_type)).to(dtype)
    target_model.set_capture_layers(draft_model.target_layer_ids)

    print_on_rank0(
        f"Draft config: block_size={draft_config.block_size}, "
        f"num_hidden_layers={draft_config.num_hidden_layers}, "
        f"target_layer_ids={draft_model.target_layer_ids}"
    )
    print_on_rank0(
        f"Draft model parameters: {sum(p.numel() for p in draft_model.parameters()):,}"
    )
    return target_model, draft_model


def build_dataloaders(args, processor, tokenizer) -> Tuple[DataLoader, Optional[DataLoader]]:
    cache_params_string = (
        f"{args.train_data_path}-{args.image_root}-{args.max_length}-"
        f"{args.chat_template}-{args.target_model_path}-{args.min_pixels}-{args.max_pixels}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    train_dataset = build_eagle3_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        is_preformatted=args.is_preformatted,
        cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=cache_key,
        num_proc=args.build_dataset_num_proc,
        is_vlm=True,
        processor=processor,
        image_root=args.image_root,
        train_only_last_turn=args.train_only_last_turn,
    )

    min_loss_tokens = 2 * args.block_size
    original_size = len(train_dataset)
    train_dataset = train_dataset.filter(lambda x: x["loss_mask"].sum() >= min_loss_tokens)
    print_on_rank0(
        f"Filtered train dataset: {original_size} -> {len(train_dataset)} samples"
    )

    train_dataloader = prepare_dp_dataloaders(
        train_dataset,
        args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=get_dp_group(),
        is_vlm=True,
    )

    eval_dataloader = None
    if args.eval_data_path:
        eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
        eval_dataset = build_eagle3_dataset(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            is_preformatted=args.is_preformatted,
            num_proc=args.build_dataset_num_proc,
            is_vlm=True,
            processor=processor,
            image_root=args.image_root,
            train_only_last_turn=args.train_only_last_turn,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_dataset,
            args.batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle=False,
            process_group=get_dp_group(),
            is_vlm=True,
        )

    return train_dataloader, eval_dataloader


def resolve_mask_token_id(args, tokenizer, draft_model, target_components) -> int:
    dflash_config = getattr(draft_model.config, "dflash_config", {}) or {}
    if args.mask_token_id is not None:
        mask_token_id = args.mask_token_id
    elif "mask_token_id" in dflash_config:
        mask_token_id = dflash_config["mask_token_id"]
    elif getattr(tokenizer, "mask_token_id", None) is not None:
        mask_token_id = tokenizer.mask_token_id
    else:
        raise ValueError(
            "No mask token id found. Provide --mask-token-id or set "
            "dflash_config.mask_token_id in the draft config."
        )

    vocab_size = target_components.embed_tokens.num_embeddings
    if mask_token_id >= vocab_size:
        raise ValueError(
            f"mask_token_id={mask_token_id} is outside target embedding vocab "
            f"size {vocab_size}."
        )
    return int(mask_token_id)


def save_checkpoint(args, epoch, step, dflash_model, draft_model, optimizer):
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

            modeling_src = os.path.join(
                os.path.dirname(__file__),
                "..",
                "specforge",
                "modeling",
                "draft",
                "dflash.py",
            )
            if os.path.exists(modeling_src):
                shutil.copy(modeling_src, os.path.join(save_dir, "dflash.py"))

            print_on_rank0(f"Saved checkpoint to {save_dir}")

    dist.barrier()


def record_metrics(args, loss, accuracy, global_step, tracker, optimizer, train_dataloader):
    tracker.log(
        {
            "train/lr": optimizer.get_learning_rate(),
            "train/loss": loss,
            "train/accuracy": accuracy,
        },
        step=global_step,
    )
    print_on_rank0(
        f"Train - Step {global_step} "
        f"[{global_step}/{args.num_epochs * len(train_dataloader) // args.accumulation_steps}?], "
        f"Loss: {loss:.4f}, Acc: {accuracy:.4f}"
    )


def move_batch_to_device(data, args, dtype: torch.dtype):
    device = current_device(args.device_type)
    batch = {
        "input_ids": data["input_ids"].to(device),
        "attention_mask": data["attention_mask"].to(device),
        "loss_mask": data["loss_mask"].to(device),
        "pixel_values": data["pixel_values"].to(device=device, dtype=dtype),
        "image_grid_thw": data["image_grid_thw"].to(device),
    }
    if "position_ids" in data:
        batch["position_ids"] = data["position_ids"].to(device)
    else:
        batch["position_ids"] = None
    return batch


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
    dtype = parse_dtype(args.dtype)
    set_seed(args.seed)

    init_distributed(
        timeout=args.dist_timeout,
        tp_size=args.tp_size,
        backend=args.dist_backend,
        device_type=args.device_type,
    )
    print_with_rank("Initialized distributed")

    draft_model_last_checkpoint = None
    ckpt_info = (0, 0)
    if args.resume and os.path.isdir(args.output_dir):
        draft_model_last_checkpoint, ckpt_info = get_last_checkpoint(args.output_dir)
        print_on_rank0(f"Last checkpoint detected: {draft_model_last_checkpoint}")

    if draft_model_last_checkpoint:
        checkpoint_config_path = os.path.join(draft_model_last_checkpoint, "config.json")
        if os.path.exists(checkpoint_config_path):
            args.draft_config_path = checkpoint_config_path

    processor, tokenizer = build_processor_and_tokenizer(args)
    target_model, draft_model = build_models(args, dtype=dtype)

    resume_state = None
    if draft_model_last_checkpoint:
        loaded_model = DFlashDraftModel.from_pretrained(
            draft_model_last_checkpoint,
            torch_dtype=dtype,
        )
        draft_model.load_state_dict(loaded_model.state_dict())
        del loaded_model
        print_on_rank0("Loaded draft model weights from checkpoint")

        training_state_path = os.path.join(
            draft_model_last_checkpoint, "training_state.pt"
        )
        if os.path.exists(training_state_path):
            resume_state = torch.load(
                training_state_path,
                map_location="cpu",
                weights_only=False,
            )

    print_on_rank0("Loading target text embeddings and LM head...")
    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key=args.embedding_key,
        lm_head_key=args.lm_head_key,
        cache_dir=args.model_download_dir,
        device=args.device_type,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )

    mask_token_id = resolve_mask_token_id(args, tokenizer, draft_model, target_components)
    draft_model.mask_token_id = mask_token_id
    draft_model.config.dflash_config["mask_token_id"] = mask_token_id
    draft_model.config.dflash_config["target_layer_ids"] = draft_model.target_layer_ids
    print_on_rank0(f"Using mask_token_id: {mask_token_id}")
    print_on_rank0(f"dflash_config: {draft_model.config.dflash_config}")

    train_dataloader, eval_dataloader = build_dataloaders(args, processor, tokenizer)
    _ = eval_dataloader

    steps_per_epoch = math.ceil(len(train_dataloader) / args.accumulation_steps)
    total_steps = args.num_epochs * steps_per_epoch
    print_on_rank0(f"Total training steps: {total_steps}")

    dflash_model = OnlineDFlashModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=draft_model.block_size,
        mask_token_id=mask_token_id,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        loss_type=args.loss_type,
        dpace_alpha=args.dpace_alpha,
    )

    fsdp_kwargs = dict(
        use_orig_params=True,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        mixed_precision=MixedPrecision(param_dtype=dtype, buffer_dtype=dtype),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    block_names = set(getattr(draft_model, "_no_split_modules", None) or [])
    block_classes = {
        type(m) for m in dflash_model.modules() if type(m).__name__ in block_names
    }
    if block_classes:
        fsdp_kwargs["auto_wrap_policy"] = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=block_classes,
        )
    dflash_model = FSDP(dflash_model, **fsdp_kwargs)
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
            f"Restored optimizer/scheduler state: epoch={start_epoch}, "
            f"step={global_step}, lr={optimizer.get_learning_rate():.6f}"
        )

    skip_steps = global_step - start_epoch * len(train_dataloader)
    tracker = create_tracker(args, args.output_dir)
    last_time = time.time()
    print_on_rank0(f"Starting training from epoch {start_epoch}, step {global_step}")

    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        draft_model.train()
        progress_bar = (
            tqdm(train_dataloader, desc=f"Training Epoch {epoch}", leave=True)
            if dist.get_rank() == 0
            else train_dataloader
        )

        for step_in_epoch, data in enumerate(progress_bar):
            if epoch == start_epoch and step_in_epoch < skip_steps:
                continue
            global_step += 1

            batch = move_batch_to_device(data, args, dtype=dtype)
            target_output = target_model.generate_dflash_data(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                loss_mask=batch["loss_mask"],
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
                position_ids=batch["position_ids"],
            )

            loss, accuracy = dflash_model(
                input_ids=batch["input_ids"],
                hidden_states=target_output.hidden_states.to(current_device(args.device_type)),
                loss_mask=batch["loss_mask"],
            )
            (loss / args.accumulation_steps).backward()

            if global_step % args.accumulation_steps == 0:
                optimizer.step()

            if global_step % args.log_interval == 0:
                loss_log = loss.detach().clone()
                acc_log = accuracy.detach().clone()
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
