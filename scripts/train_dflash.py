#!/usr/bin/env python3
# coding=utf-8
"""DFlash Training Script."""

import argparse
import copy
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
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from datasets import load_dataset
from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.core.dflash import OnlineDFlashModel, QwenVLOnlineDFlashModel
from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.target.dflash_target_model import (
    DFlashTargetModel,
    HFDFlashTargetModel,
    get_dflash_target_model,
)
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker
from specforge.utils import (
    get_last_checkpoint,
    print_on_rank0,
    print_with_rank,
)

QWEN3_VL_MODEL_TYPES = {"qwen3_vl", "qwen3_vl_moe"}


def _resolve_target_num_hidden_layers(target_config) -> int:
    # For VLM configs (e.g. Qwen3-VL), top-level num_hidden_layers may refer to
    # vision stack depth. DFlash target layers must follow language model depth.
    if hasattr(target_config, "text_config") and hasattr(
        target_config.text_config, "num_hidden_layers"
    ):
        return target_config.text_config.num_hidden_layers
    if hasattr(target_config, "num_hidden_layers"):
        return target_config.num_hidden_layers
    raise ValueError(
        f"Cannot infer num_target_layers from config type {type(target_config)}"
    )


def _build_target_layer_ids(
    num_target_layers: int,
    num_draft_layers: int,
    start_layer: int = 1,
    end_layer: Optional[int] = None,
) -> list[int]:
    """Build evenly spaced target layer ids."""
    if num_draft_layers <= 0:
        raise ValueError("num_draft_layers must be positive.")

    if end_layer is None:
        end_layer = num_target_layers - 3

    max_layer_idx = num_target_layers - 1
    start_layer = max(0, min(start_layer, max_layer_idx))
    end_layer = max(0, min(end_layer, max_layer_idx))

    if end_layer < start_layer:
        raise ValueError(
            f"Invalid layer range: start_layer={start_layer}, end_layer={end_layer}"
        )

    if num_draft_layers == 1:
        midpoint = num_target_layers // 2
        return [max(start_layer, min(midpoint, end_layer))]

    span = end_layer - start_layer
    return [
        int(start_layer + (i * span) / (num_draft_layers - 1))
        for i in range(num_draft_layers)
    ]


def _resolve_draft_config(target_config):
    model_type = getattr(target_config, "model_type", None)
    if model_type in QWEN3_VL_MODEL_TYPES and hasattr(target_config, "text_config"):
        draft_config = copy.deepcopy(target_config.text_config)
        for attr_name in (
            "dflash_config",
            "block_size",
            "rope_scaling",
            "rope_theta",
            "max_position_embeddings",
        ):
            if not hasattr(target_config, attr_name):
                continue
            if not hasattr(draft_config, attr_name) or getattr(
                draft_config, attr_name
            ) is None:
                setattr(draft_config, attr_name, getattr(target_config, attr_name))
        return draft_config
    return copy.deepcopy(target_config)


def _resolve_target_weight_keys(target_config) -> Tuple[str, str]:
    model_type = getattr(target_config, "model_type", None)
    if model_type in QWEN3_VL_MODEL_TYPES:
        return "model.language_model.embed_tokens.weight", "lm_head.weight"
    return "model.embed_tokens.weight", "lm_head.weight"


def _ensure_layer_types(draft_config) -> None:
    if hasattr(draft_config, "layer_types") and draft_config.layer_types is not None:
        return

    if not hasattr(draft_config, "num_hidden_layers"):
        return

    num_hidden_layers = draft_config.num_hidden_layers
    sliding_window = getattr(draft_config, "sliding_window", None)
    max_window_layers = getattr(draft_config, "max_window_layers", num_hidden_layers)
    if max_window_layers is None:
        max_window_layers = num_hidden_layers
    draft_config.layer_types = [
        "sliding_attention"
        if sliding_window is not None and layer_idx >= max_window_layers
        else "full_attention"
        for layer_idx in range(num_hidden_layers)
    ]


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
    model_group.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code"
    )
    model_group.add_argument(
        "--is-vlm",
        action="store_true",
        help="Whether to enable VLM training mode. If not set, will auto-detect from target model config.",
    )
    model_group.add_argument(
        "--num-anchors",
        type=int,
        default=512,
        help="Number of anchor positions per sequence",
    )
    model_group.add_argument(
        "--loss-decay-gamma",
        type=float,
        default=None,
        help="Gamma for exponential loss decay weighting (paper Eq.4). "
        "Suggested: 7 for block_size=16, 5 for 10, 4 for 8. None disables.",
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
    dataset_group.add_argument(
        "--min-pixels",
        type=int,
        default=50176,
        help="Minimum image pixels for VLM processor.",
    )
    dataset_group.add_argument(
        "--max-pixels",
        type=int,
        default=802816,
        help="Maximum image pixels for VLM processor.",
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
    training_group.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Directory of the checkpoint to resume training from",
    )

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

    # SGLang specific args
    sglang_group = parser.add_argument_group("sglang backend")
    SGLangBackendArgs.add_args(sglang_group)

    return parser.parse_args()


def build_models(
    args, target_config=None, is_vlm: bool = False
) -> Tuple[DFlashTargetModel, DFlashDraftModel, AutoConfig]:
    """Build target model (backend wrapper) and draft model."""
    print_on_rank0(
        f"Loading target model from {args.target_model_path} using {args.target_model_backend} backend"
    )

    if target_config is None:
        target_config = AutoConfig.from_pretrained(
            args.target_model_path, trust_remote_code=args.trust_remote_code
        )
    target_model_type = getattr(target_config, "model_type", None)

    if (
        args.target_model_backend == "hf"
        and is_vlm
        and target_model_type == "qwen3_vl"
        and args.tp_size == 1
    ):
        from transformers import Qwen3VLForConditionalGeneration

        target_model = HFDFlashTargetModel(
            Qwen3VLForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=args.target_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=args.trust_remote_code,
            )
            .eval()
            .cuda(),
            model_type=target_model_type,
        )
    elif (
        args.target_model_backend == "hf"
        and is_vlm
        and target_model_type == "qwen3_vl_moe"
        and args.tp_size == 1
    ):
        from transformers import Qwen3VLMoeForConditionalGeneration

        target_model = HFDFlashTargetModel(
            Qwen3VLMoeForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=args.target_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=args.trust_remote_code,
            )
            .eval()
            .cuda(),
            model_type=target_model_type,
        )
    else:
        target_model_kwargs = {}
        if args.target_model_backend == "sglang":
            target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()

        target_model = get_dflash_target_model(
            pretrained_model_name_or_path=args.target_model_path,
            backend=args.target_model_backend,
            torch_dtype=torch.bfloat16,
            device="cuda" if args.target_model_backend == "hf" else None,
            trust_remote_code=args.trust_remote_code,
            **target_model_kwargs,
        )

    # Resolve before draft config mutations to avoid reading modified values.
    target_num_layers = _resolve_target_num_hidden_layers(target_config)

    if args.draft_config_path:
        draft_config = AutoConfig.from_pretrained(
            args.draft_config_path, trust_remote_code=args.trust_remote_code
        )
        draft_config = _resolve_draft_config(draft_config)
        print_on_rank0(f"Loaded draft config from {args.draft_config_path}")
    else:
        draft_config = _resolve_draft_config(target_config)
        draft_config.num_hidden_layers = args.num_draft_layers
        draft_config.block_size = args.block_size
        print_on_rank0("Auto-generated draft config from target model")

    # Always use target language model depth for capture layer mapping.
    draft_config.num_target_layers = target_num_layers
    _ensure_layer_types(draft_config)

    if not hasattr(draft_config, "dflash_config") or draft_config.dflash_config is None:
        draft_config.dflash_config = {}
    elif not isinstance(draft_config.dflash_config, dict):
        draft_config.dflash_config = dict(draft_config.dflash_config)

    model_type = getattr(target_config, "model_type", None)
    if model_type in QWEN3_VL_MODEL_TYPES:
        # Keep the original evenly spaced mapping, but force the first capture
        # layer to skip Qwen3-VL deepstack layers (0-2).
        recommended_layer_ids = _build_target_layer_ids(
            num_target_layers=target_num_layers,
            num_draft_layers=draft_config.num_hidden_layers,
        )
        if recommended_layer_ids and recommended_layer_ids[0] < 3:
            recommended_layer_ids[0] = 3
        if "target_layer_ids" not in draft_config.dflash_config:
            draft_config.dflash_config["target_layer_ids"] = recommended_layer_ids
            print_on_rank0(
                "Qwen3-VL detected: default target_layer_ids set to "
                f"{draft_config.dflash_config['target_layer_ids']} "
                "(first layer forced to 3)."
            )
        elif (
            draft_config.dflash_config["target_layer_ids"]
            and draft_config.dflash_config["target_layer_ids"][0] < 3
        ):
            old_layer_ids = draft_config.dflash_config["target_layer_ids"]
            new_layer_ids = list(old_layer_ids)
            new_layer_ids[0] = 3
            draft_config.dflash_config["target_layer_ids"] = new_layer_ids
            print_on_rank0(
                "Qwen3-VL detected: overriding first target layer "
                f"{old_layer_ids} -> {new_layer_ids} "
                "to avoid deepstack train/serve mismatch."
            )

    draft_config._attn_implementation = args.attention_backend
    print_on_rank0(f"Using attention backend: {args.attention_backend}")

    draft_model = DFlashDraftModel(draft_config).cuda().to(torch.bfloat16)

    target_model.set_capture_layers(draft_model.target_layer_ids)

    print_on_rank0(
        f"Draft config: block_size={draft_config.block_size}, "
        f"num_hidden_layers={draft_config.num_hidden_layers}, "
        f"num_target_layers={draft_config.num_target_layers}"
    )
    print_on_rank0(
        f"Draft model parameters: {sum(p.numel() for p in draft_model.parameters()):,}"
    )

    return target_model, draft_model, target_config


def build_dataloader(
    args,
    tokenizer,
    is_vlm: bool = False,
    processor: Optional[AutoProcessor] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build train and eval dataloaders."""
    import hashlib

    cache_params_string = (
        f"{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
    cache_dir = os.path.join(args.cache_dir, "processed_dataset")
    dist_enabled = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if dist_enabled else 0
    world_size = dist.get_world_size() if dist_enabled else 1

    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    if world_size > 1:
        if rank == 0:
            train_eagle3_dataset = build_eagle3_dataset(
                dataset=train_dataset,
                tokenizer=tokenizer,
                chat_template=args.chat_template,
                max_length=args.max_length,
                is_preformatted=args.is_preformatted,
                is_vlm=is_vlm,
                processor=processor,
                cache_dir=cache_dir,
                cache_key=cache_key,
                num_proc=args.build_dataset_num_proc,
            )
            dist.barrier()
        else:
            dist.barrier()
            train_eagle3_dataset = build_eagle3_dataset(
                dataset=train_dataset,
                tokenizer=tokenizer,
                chat_template=args.chat_template,
                max_length=args.max_length,
                is_preformatted=args.is_preformatted,
                is_vlm=is_vlm,
                processor=processor,
                cache_dir=cache_dir,
                cache_key=cache_key,
                # Rank 0 has finished preprocessing at this point; other ranks only need cache reads.
                num_proc=1,
            )
    else:
        train_eagle3_dataset = build_eagle3_dataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            is_preformatted=args.is_preformatted,
            is_vlm=is_vlm,
            processor=processor,
            cache_dir=cache_dir,
            cache_key=cache_key,
            num_proc=args.build_dataset_num_proc,
        )

    min_loss_tokens = 2 * args.block_size
    original_size = len(train_eagle3_dataset)
    train_eagle3_dataset = train_eagle3_dataset.filter(
        lambda x: x["loss_mask"].sum() >= min_loss_tokens
    )
    print_on_rank0(
        f"Filtered train dataset: {original_size} -> {len(train_eagle3_dataset)} samples"
    )

    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=get_dp_group(),
        is_vlm=is_vlm,
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
            is_vlm=is_vlm,
            processor=processor,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle=False,
            process_group=get_dp_group(),
            is_vlm=is_vlm,
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

            modeling_src = os.path.join(
                os.path.dirname(__file__),
                "..",
                "specforge",
                "modeling",
                "draft",
                "dflash.py",
            )
            modeling_dst = os.path.join(save_dir, "dflash.py")
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

    target_config = AutoConfig.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    detected_vlm = getattr(target_config, "model_type", None) in QWEN3_VL_MODEL_TYPES
    is_vlm = args.is_vlm or detected_vlm
    if detected_vlm and not args.is_vlm:
        print_on_rank0(
            "Detected Qwen3-VL target config; enabling VLM mode automatically."
        )
    if is_vlm and args.target_model_backend != "hf":
        raise ValueError(
            "Real multimodal DFlash training currently supports only HF backend. "
            "Please set --target-model-backend hf."
        )
    print_on_rank0(
        f"Detected target model_type={getattr(target_config, 'model_type', None)}, is_vlm={is_vlm}"
    )

    target_model, draft_model, target_config = build_models(
        args, target_config, is_vlm=is_vlm
    )

    draft_model_last_checkpoint = None
    if args.ckpt_dir is not None:
        if os.path.isdir(args.ckpt_dir):
            draft_model_last_checkpoint = args.ckpt_dir
            print_on_rank0(f"Using checkpoint: {draft_model_last_checkpoint}")
        else:
            raise ValueError(
                f"Provided ckpt dir {args.ckpt_dir} is not a valid directory."
            )

    if args.resume and os.path.isdir(args.output_dir):
        draft_model_last_checkpoint = get_last_checkpoint(
            args.output_dir, prefix=r"epoch_\d+_step"
        )
        print_on_rank0(f"Last checkpoint detected: {draft_model_last_checkpoint}")

    resume_state = None
    if draft_model_last_checkpoint:
        loaded_model = DFlashDraftModel.from_pretrained(
            draft_model_last_checkpoint, torch_dtype=torch.bfloat16
        )
        draft_model.load_state_dict(loaded_model.state_dict())
        del loaded_model
        print_on_rank0("Loaded draft model weights from checkpoint")

        training_state_path = os.path.join(
            draft_model_last_checkpoint, "training_state.pt"
        )
        if os.path.exists(training_state_path):
            resume_state = torch.load(
                training_state_path, map_location="cpu", weights_only=False
            )
            print_on_rank0(
                f"Will resume from epoch {resume_state['epoch']}, "
                f"step {resume_state['global_step']}"
            )

    if is_vlm:
        processor = AutoProcessor.from_pretrained(
            args.target_model_path,
            trust_remote_code=args.trust_remote_code,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        tokenizer = processor.tokenizer
    else:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    if args.mask_token_id is not None:
        mask_token_id = args.mask_token_id
    elif tokenizer.mask_token_id is not None:
        mask_token_id = tokenizer.mask_token_id
    else:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        mask_token_id = tokenizer.mask_token_id
    print_on_rank0(f"Using mask_token_id: {mask_token_id}")

    draft_model.mask_token_id = mask_token_id
    draft_model.config.dflash_config["mask_token_id"] = mask_token_id
    draft_model.config.dflash_config["target_layer_ids"] = draft_model.target_layer_ids
    print_on_rank0(f"dflash_config: {draft_model.config.dflash_config}")

    train_dataloader, eval_dataloader = build_dataloader(
        args,
        tokenizer,
        is_vlm=is_vlm,
        processor=processor,
    )

    steps_per_epoch = math.ceil(len(train_dataloader) / args.accumulation_steps)
    total_steps = args.num_epochs * steps_per_epoch
    print_on_rank0(f"Total training steps: {total_steps}")

    print_on_rank0("Loading target embeddings and head...")
    embed_key, lm_head_key = _resolve_target_weight_keys(target_config)
    print_on_rank0(
        f"Loading target embeddings/head with keys: embed='{embed_key}', head='{lm_head_key}'"
    )
    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key=embed_key,
        lm_head_key=lm_head_key,
        device="cuda",
        trust_remote_code=args.trust_remote_code,
    )

    dflash_model_cls = OnlineDFlashModel
    if getattr(target_config, "model_type", None) == "qwen3_vl":
        dflash_model_cls = QwenVLOnlineDFlashModel
    print_on_rank0(f"Using DFlash wrapper: {dflash_model_cls.__name__}")
    dflash_model = dflash_model_cls(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=draft_model.block_size,
        mask_token_id=mask_token_id,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
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

    start_epoch = 0
    global_step = 0
    if resume_state is not None:
        optimizer.scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        del resume_state
        print_on_rank0(f"Restored scheduler, lr={optimizer.get_learning_rate():.6f}")

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

            input_ids = data["input_ids"].cuda()
            attention_mask = data["attention_mask"].cuda()
            loss_mask = data["loss_mask"].cuda()
            target_kwargs = {}
            if is_vlm:
                if "pixel_values" in data:
                    target_kwargs["pixel_values"] = data["pixel_values"].cuda()
                if "pixel_values_videos" in data:
                    target_kwargs["pixel_values_videos"] = data["pixel_values_videos"].cuda()
                if "image_grid_thw" in data:
                    target_kwargs["image_grid_thw"] = data["image_grid_thw"].cuda()
                if "video_grid_thw" in data:
                    target_kwargs["video_grid_thw"] = data["video_grid_thw"].cuda()
                if "second_per_grid_ts" in data:
                    target_kwargs["second_per_grid_ts"] = data["second_per_grid_ts"].cuda()
            target_output = target_model.generate_dflash_data(
                input_ids, attention_mask, loss_mask, **target_kwargs
            )
            hidden_states = target_output.hidden_states.cuda()  # Ensure on GPU
            position_ids = (
                target_output.position_ids.cuda()
                if target_output.position_ids is not None
                else None
            )

            loss, accuracy = dflash_model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
                position_ids=position_ids,
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
