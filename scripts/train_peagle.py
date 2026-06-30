"""P-EAGLE (Parallel EAGLE) training script.

Based on train_eagle3.py but replaces TTT with COD parallel sampling.
"""

import argparse
import hashlib
import json
import math
import os
import time
from argparse import ArgumentParser, Namespace
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import set_seed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from datasets import DatasetDict, load_dataset
from specforge import AutoDraftModelConfig, get_eagle3_target_model
from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.core.peagle import OnlinePEagleModel
from specforge.data import (
    build_eagle3_dataset,
    generate_vocab_mapping_file,
    prepare_dp_dataloaders,
)
from specforge.distributed import (
    destroy_distributed,
    get_dp_group,
    get_tp_group,
    init_distributed,
)
from specforge.modeling.draft.peagle import PEagleDraftModel
from specforge.modeling.target import Eagle3TargetModel
from specforge.optimizer import BF16Optimizer
from specforge.tracker import Tracker, create_tracker, get_tracker_class
from specforge.utils import (
    get_last_checkpoint,
    print_args_with_dots,
    print_on_rank0,
    print_with_rank,
    rank_0_priority,
)


def parse_args() -> Tuple[ArgumentParser, Namespace]:
    parser = argparse.ArgumentParser(description="Train P-EAGLE with online data")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code"
    )
    model_group.add_argument("--draft-model-config", type=str, required=False)
    model_group.add_argument(
        "--embedding-key",
        type=str,
        default="model.embed_tokens.weight",
    )
    model_group.add_argument(
        "--target-model-backend",
        type=str,
        default="sglang",
        choices=["sglang", "hf", "custom"],
    )

    # P-EAGLE specific args
    peagle_group = parser.add_argument_group("peagle")
    peagle_group.add_argument(
        "--num-depths",
        type=int,
        default=8,
        help="Number of parallel prediction depths for P-EAGLE COD sampling",
    )
    peagle_group.add_argument(
        "--down-sample-ratio",
        type=float,
        default=0.8,
        help="Geometric decay ratio for COD sampling",
    )
    peagle_group.add_argument(
        "--down-sample-ratio-min",
        type=float,
        default=0.2,
        help="Minimum retention ratio for COD sampling",
    )
    peagle_group.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help="Token ID for masking. If None, uses tokenizer.pad_token_id or 0",
    )
    peagle_group.add_argument(
        "--num-draft-layers",
        type=int,
        default=4,
        help="Number of decoder layers in the P-EAGLE draft model",
    )
    peagle_group.add_argument(
        "--norm-before-residual",
        action="store_true",
        help="Whether to use normalized hidden as residual in the first layer",
    )
    peagle_group.add_argument(
        "--no-norm-before-residual",
        action="store_true",
        help="Explicitly disable norm-before-residual",
    )

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train-data-path", type=str, required=True)
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument("--chat-template", type=str, default="llama3")
    dataset_group.add_argument("--is-preformatted", action="store_true")
    dataset_group.add_argument("--train-only-last-turn", action="store_true")
    dataset_group.add_argument("--build-dataset-num-proc", type=int, default=8)
    dataset_group.add_argument("--dataloader-num-workers", type=int, default=4)

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=10)
    training_group.add_argument("--max-num-steps", type=int, default=None)
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=6e-4)
    training_group.add_argument("--max-length", type=int, default=2048)
    training_group.add_argument("--warmup-ratio", type=float, default=0.015)
    training_group.add_argument("--total-steps", type=int, default=None)
    training_group.add_argument("--max-grad-norm", type=float, default=0.5)
    training_group.add_argument("--resume", action="store_true")
    training_group.add_argument("--ckpt-dir", type=str, default=None)
    training_group.add_argument("--eval-interval", type=int, default=5000)
    training_group.add_argument("--save-interval", type=int, default=5000)
    training_group.add_argument("--log-interval", type=int, default=50)
    training_group.add_argument("--seed", type=int, default=0)
    training_group.add_argument("--draft-accumulation-steps", type=int, default=1)

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument("--tp-size", type=int, default=1)

    other_group = parser.add_argument_group("others")
    other_group.add_argument("--cache-key", type=str, default=None)
    other_group.add_argument("--cache-dir", type=str, default="./cache")
    other_group.add_argument("--output-dir", type=str, required=True)
    other_group.add_argument("--verbose", action="store_true")
    other_group.add_argument("--dist-timeout", type=int, default=20)
    other_group.add_argument("--model-download-dir", type=str, default=None)

    profiling_group = parser.add_argument_group("profiling")
    profiling_group.add_argument("--profile", action="store_true")
    profiling_group.add_argument("--profile-start-step", type=int, default=30)
    profiling_group.add_argument("--profile-num-steps", type=int, default=4)
    profiling_group.add_argument("--profile-record-shapes", action="store_true")

    sglang_group = parser.add_argument_group("sglang target model backend")
    SGLangBackendArgs.add_args(sglang_group)

    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)

    args = parser.parse_args()
    return parser, args


def build_tracker(args: Namespace, parser: ArgumentParser) -> Tracker:
    tracker_class = get_tracker_class(args.report_to)
    if tracker_class:
        tracker_class.validate_args(parser, args)
    else:
        parser.error(f"Unknown tracker: {args.report_to}")
    return create_tracker(args, args.output_dir)


def build_target_model(
    args: Namespace, draft_model_config: AutoDraftModelConfig
) -> Eagle3TargetModel:
    if args.target_model_backend == "sglang":
        target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()
    else:
        target_model_kwargs = {}
    target_model = get_eagle3_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device="cuda",
        cache_dir=args.model_download_dir,
        **target_model_kwargs,
        trust_remote_code=args.trust_remote_code,
    )
    if (
        hasattr(draft_model_config, "eagle_config")
        and draft_model_config.eagle_config is not None
        and "eagle_aux_hidden_state_layer_ids" in draft_model_config.eagle_config
    ):
        target_model.set_aux_hidden_states_layers(
            draft_model_config.eagle_config["eagle_aux_hidden_state_layer_ids"]
        )
    else:
        target_model.set_aux_hidden_states_layers()
    return target_model


def build_draft_model(args: Namespace) -> Tuple:
    ckpt_info = (0, 0)
    resume_state = None
    should_load_target_embedding = True

    if args.draft_model_config is not None:
        draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)
    else:
        from specforge.utils import create_draft_config_from_target

        auto_config_path = create_draft_config_from_target(
            target_model_path=args.target_model_path,
            cache_dir=args.model_download_dir,
        )
        draft_model_config = AutoDraftModelConfig.from_file(auto_config_path)

    # Override num_hidden_layers for P-EAGLE multi-layer
    draft_model_config.num_hidden_layers = args.num_draft_layers

    draft_model_last_checkpoint = None
    is_resume_checkpoint = False
    if args.ckpt_dir is not None:
        if os.path.isdir(args.ckpt_dir):
            draft_model_config = AutoDraftModelConfig.from_file(
                os.path.join(args.ckpt_dir, "config.json")
            )
            draft_model_config.num_hidden_layers = args.num_draft_layers
            draft_model_last_checkpoint = args.ckpt_dir
            should_load_target_embedding = False
            print_on_rank0(f"Finetuning from base model: {draft_model_last_checkpoint}")
        else:
            raise ValueError(
                f"Provided base model dir {args.ckpt_dir} is not a valid directory."
            )

    if args.resume and os.path.isdir(args.output_dir):
        draft_model_last_checkpoint, ckpt_info = get_last_checkpoint(args.output_dir)
        print(f"Last checkpoint detected: {draft_model_last_checkpoint}")
        is_resume_checkpoint = True
        should_load_target_embedding = False

    norm_before_residual = (
        args.norm_before_residual and not args.no_norm_before_residual
    )

    if draft_model_last_checkpoint:
        draft_model = PEagleDraftModel(
            config=draft_model_config,
            norm_before_residual=norm_before_residual,
        ).to(dtype=torch.bfloat16, device="cuda")
        safetensors_path = os.path.join(
            draft_model_last_checkpoint, "model.safetensors"
        )
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file

            state_dict = load_file(safetensors_path, device="cuda")
            draft_model.load_state_dict(state_dict, strict=False)
            if "embed_tokens.weight" not in state_dict:
                should_load_target_embedding = True
                print_on_rank0(
                    "Checkpoint does not contain trainable P-EAGLE embeddings; "
                    "loading embeddings from the target model."
                )
        else:
            should_load_target_embedding = True
            print_on_rank0(
                f"No model.safetensors found in {draft_model_last_checkpoint}; "
                "loading embeddings from the target model."
            )
    else:
        draft_model = PEagleDraftModel(
            config=draft_model_config,
            norm_before_residual=norm_before_residual,
        ).to(dtype=torch.bfloat16, device="cuda")

    if is_resume_checkpoint and draft_model_last_checkpoint:
        training_state_path = os.path.join(
            draft_model_last_checkpoint, "training_state.pt"
        )
        if os.path.exists(training_state_path):
            resume_state = torch.load(
                training_state_path, map_location="cpu", weights_only=False
            )
            print_on_rank0(
                f"Loaded training state from {training_state_path}: "
                f"epoch={resume_state['epoch']}, step={resume_state['global_step']}"
            )

    if should_load_target_embedding:
        draft_model.load_embedding(
            args.target_model_path, embedding_key=args.embedding_key
        )
    else:
        print_on_rank0("Using embeddings from the P-EAGLE checkpoint.")
    return draft_model_config, draft_model, ckpt_info, resume_state


def load_conversation_dataset(data_path: str):
    """Load local JSON/JSONL data like DFlash, or an HF dataset id."""
    if os.path.isfile(data_path) and os.path.splitext(data_path)[1].lower() in (
        ".json",
        ".jsonl",
    ):
        return load_dataset("json", data_files=data_path)["train"]

    dataset = load_dataset(data_path, split="train")
    if isinstance(dataset, DatasetDict):
        if "train" not in dataset:
            raise ValueError(
                f"Expected a 'train' split, but found splits: {list(dataset.keys())}"
            )
        return dataset["train"]
    return dataset


def build_dataloaders(
    args: Namespace,
    draft_model_config,
) -> Tuple[DataLoader, str, Optional[DataLoader]]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )

    draft_vocab_size = getattr(
        draft_model_config, "draft_vocab_size", draft_model_config.vocab_size
    )
    cache_params_string = (
        f"{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}-"
        f"{draft_vocab_size}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
    train_dataset = load_conversation_dataset(args.train_data_path)
    with rank_0_priority():
        train_eagle3_dataset = build_eagle3_dataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=cache_key,
            is_vlm=False,
            is_preformatted=args.is_preformatted,
            processor=None,
            num_proc=args.build_dataset_num_proc,
            train_only_last_turn=args.train_only_last_turn,
            minimum_valid_tokens=1,
        )
        vocab_mapping_path = generate_vocab_mapping_file(
            dataset=train_eagle3_dataset,
            target_vocab_size=draft_model_config.vocab_size,
            draft_vocab_size=draft_vocab_size,
            cache_dir=os.path.join(args.cache_dir, "vocab_mapping"),
            cache_key=cache_key,
        )

    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.target_batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=get_dp_group(),
        is_vlm=False,
    )

    eval_dataloader = None
    if args.eval_data_path is not None:
        eval_dataset = load_conversation_dataset(args.eval_data_path)
        eval_eagle3_dataset = build_eagle3_dataset(
            eval_dataset,
            tokenizer,
            args.chat_template,
            args.max_length,
            is_vlm=False,
            processor=None,
            num_proc=args.build_dataset_num_proc,
            is_preformatted=args.is_preformatted,
            train_only_last_turn=args.train_only_last_turn,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.target_batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle=False,
            process_group=get_dp_group(),
            is_vlm=False,
        )
        print_with_rank("Initialized eval dataloader")

    return train_dataloader, vocab_mapping_path, eval_dataloader


def save_checkpoints(
    args: Namespace,
    epoch: int,
    step: int,
    peagle_model: nn.Module,
    optimizer: Optimizer,
):
    epoch_output_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(epoch_output_dir, exist_ok=True)
    dist.barrier()

    with FSDP.state_dict_type(peagle_model, StateDictType.FULL_STATE_DICT):
        model_state_dict = peagle_model.state_dict()
        state_to_save = {
            "epoch": epoch,
            "global_step": step,
            "args": args,
        }
        state_to_save.update(optimizer.state_dict())
        draft_model_state_dict = {
            k.replace("draft_model.", ""): v
            for k, v in model_state_dict.items()
            if "draft_model." in k
        }

        if dist.get_rank() == 0:
            torch.save(
                state_to_save,
                os.path.join(epoch_output_dir, "training_state.pt"),
            )
            peagle_model.draft_model.save_pretrained(
                epoch_output_dir,
                state_dict=draft_model_state_dict,
            )
            peagle_config = {
                "num_depths": args.num_depths,
                "down_sample_ratio": args.down_sample_ratio,
                "down_sample_ratio_min": args.down_sample_ratio_min,
                "mask_token_id": args.mask_token_id,
                "num_draft_layers": args.num_draft_layers,
                "norm_before_residual": args.norm_before_residual,
            }
            with open(os.path.join(epoch_output_dir, "peagle_config.json"), "w") as f:
                json.dump(peagle_config, f, indent=2)

            print_on_rank0(f"Saved model to {epoch_output_dir}")
        dist.barrier()


def get_dp_data_shard_from_tp(tensor: torch.Tensor) -> torch.Tensor:
    tp_size = dist.get_world_size(get_tp_group())
    tp_rank = dist.get_rank(get_tp_group())
    return tensor.chunk(tp_size, dim=0)[tp_rank]


def run_forward(
    args: Namespace,
    peagle_model: nn.Module,
    data: dict,
    target_model: Eagle3TargetModel,
) -> Tuple[torch.Tensor, Dict]:
    eagle3_data = target_model.generate_eagle3_data(
        input_ids=data["input_ids"].cuda(),
        attention_mask=data["attention_mask"].cuda(),
        loss_mask=data["loss_mask"].cuda(),
    )

    input_ids = get_dp_data_shard_from_tp(eagle3_data.input_ids)
    attention_mask = get_dp_data_shard_from_tp(eagle3_data.attention_mask)
    loss_mask = get_dp_data_shard_from_tp(eagle3_data.loss_mask)
    target = get_dp_data_shard_from_tp(eagle3_data.target)
    hidden_states = get_dp_data_shard_from_tp(eagle3_data.hidden_states)

    loss, metrics = peagle_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        target=target,
        hidden_states=hidden_states,
    )
    return loss, metrics


def record_metrics(
    args: Namespace,
    metrics: Dict,
    global_step: int,
    tracker: Tracker,
    optimizer: Optional[Optimizer] = None,
    mode: str = "train",
) -> None:
    logdict = {}

    if mode == "train" and optimizer is not None:
        logdict["train/lr"] = optimizer.get_learning_rate()

    loss = metrics.get("loss_sum", torch.tensor(0.0))
    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
    logdict[f"{mode}/loss"] = loss.item()
    print_on_rank0(f"{mode} - Step {global_step}, Loss: {loss.item():.4f}")

    full_acc_sum = metrics.get("full_acc_sum", torch.tensor(0.0))
    full_acc_total = metrics.get("full_acc_total", torch.tensor(1.0))
    dist.all_reduce(full_acc_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(full_acc_total, op=dist.ReduceOp.SUM)
    full_acc = (full_acc_sum / full_acc_total.clamp_min(1)).item()
    logdict[f"{mode}/acc"] = full_acc
    print_on_rank0(f"{mode} - Step {global_step}, Acc: {full_acc:.4f}")

    for d in range(args.num_depths):
        key_sum = f"position_{d}_acc_sum"
        key_total = f"position_{d}_acc_total"
        if key_sum in metrics:
            d_sum = metrics[key_sum]
            d_total = metrics[key_total]
            dist.all_reduce(d_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(d_total, op=dist.ReduceOp.SUM)
            d_acc = (d_sum / d_total.clamp_min(1)).item()
            logdict[f"{mode}/acc_depth_{d}"] = d_acc
            print_on_rank0(f"{mode} - Step {global_step}, Depth {d} Acc: {d_acc:.4f}")

    tracker.log(logdict, step=global_step)


def _print_on_rank0_or_local(message: str) -> None:
    if dist.is_available() and dist.is_initialized():
        print_on_rank0(message)
    else:
        print_with_rank(message)


def _validate_mask_token_id(mask_token_id: int, embedding_vocab_size: int) -> int:
    if not 0 <= mask_token_id < embedding_vocab_size:
        raise ValueError(
            f"mask_token_id {mask_token_id} is outside embedding vocab "
            f"size {embedding_vocab_size}."
        )
    return mask_token_id


def resolve_mask_token_id(args: Namespace, embedding_vocab_size: int) -> int:
    if args.mask_token_id is not None:
        return _validate_mask_token_id(args.mask_token_id, embedding_vocab_size)

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    if getattr(tokenizer, "mask_token_id", None) is not None:
        mask_token_id = _validate_mask_token_id(
            tokenizer.mask_token_id, embedding_vocab_size
        )
        _print_on_rank0_or_local(
            f"Auto-set mask_token_id to tokenizer mask token {mask_token_id}"
        )
        return mask_token_id

    if len(tokenizer) < embedding_vocab_size:
        mask_token_id = len(tokenizer)
        _print_on_rank0_or_local(
            f"Auto-set mask_token_id to unused embedding slot {mask_token_id}"
        )
        return mask_token_id

    for token_name in ("pad_token_id", "eos_token_id", "unk_token_id"):
        token_id = getattr(tokenizer, token_name, None)
        if token_id is not None:
            mask_token_id = _validate_mask_token_id(token_id, embedding_vocab_size)
            _print_on_rank0_or_local(
                "Tokenizer has no mask token or unused draft embedding slot; "
                f"falling back to {token_name}={mask_token_id}. "
                "Pass --mask-token-id to use a dedicated trainable mask token."
            )
            return mask_token_id

    raise ValueError(
        "Could not resolve mask_token_id. Pass --mask-token-id or use a tokenizer "
        "with mask/pad/eos/unk token."
    )


def main():
    # ================================================
    # 1. Initialize
    # ================================================
    parser, args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)

    args.dp_size = dist.get_world_size() // args.tp_size
    args.target_batch_size = args.tp_size * args.batch_size

    print_args_with_dots(args)
    print_with_rank("Initialized distributed environment")

    # ================================================
    # 2. Build models
    # ================================================
    draft_model_config, draft_model, ckpt_info, resume_state = build_draft_model(args)
    target_model = build_target_model(args, draft_model_config)

    # ================================================
    # 3. Build dataloader
    # ================================================
    train_dataloader, vocab_mapping_path, eval_dataloader = build_dataloaders(
        args, draft_model_config
    )
    draft_model.load_vocab_mapping(vocab_mapping_path)
    print_with_rank("Loaded vocab mapping")

    # Resolve mask_token_id
    args.mask_token_id = resolve_mask_token_id(
        args,
        draft_model_config.vocab_size,
    )

    # Calculate total steps
    if args.total_steps is None:
        steps_per_epoch = math.ceil(
            len(train_dataloader) / args.draft_accumulation_steps
        )
        args.total_steps = args.num_epochs * steps_per_epoch
        print_with_rank(f"Auto-calculated total_steps: {args.total_steps}")

    # ================================================
    # 4. Build P-EAGLE model
    # ================================================
    peagle_model = OnlinePEagleModel(
        draft_model=draft_model,
        mask_token_id=args.mask_token_id,
        num_depths=args.num_depths,
        down_sample_ratio=args.down_sample_ratio,
        down_sample_ratio_min=args.down_sample_ratio_min,
    )

    # ================================================
    # 5. Wrap with FSDP, then build optimizer and scheduler
    # ================================================
    peagle_model = FSDP(
        peagle_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        process_group=dist.group.WORLD,
        device_id=torch.cuda.current_device(),
    )

    # Build optimizer after FSDP so fp32 param copies match sharded shapes
    optimizer = BF16Optimizer(
        peagle_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=args.total_steps,
    )
    print_with_rank("Initialized optimizer and scheduler")

    if resume_state is not None:
        optimizer.load_state_dict(resume_state)
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        print_on_rank0(
            f"Restored optimizer/scheduler state: "
            f"epoch={start_epoch}, step={global_step}, "
            f"lr={optimizer.get_learning_rate():.6f}"
        )
        del resume_state
    else:
        start_epoch = ckpt_info[0]
        global_step = ckpt_info[1]

    skip_steps = global_step - start_epoch * len(train_dataloader)

    # ================================================
    # 6. Build tracker
    # ================================================
    tracker = build_tracker(args, parser)
    dist.barrier()

    last_time = time.time()

    # ================================================
    # 7. Start training
    # ================================================
    print_on_rank0(
        f"Starting P-EAGLE training from epoch:{start_epoch} step:{global_step}"
    )

    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch + 1)
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

            # Profiling
            if args.profile:
                if global_step == args.profile_start_step + 1:
                    print("Start profile")
                    torch_profiler = torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        with_stack=True,
                        record_shapes=args.profile_record_shapes,
                    )
                    torch_profiler.start()
                if global_step == args.profile_start_step + args.profile_num_steps + 1:
                    output_path = os.path.join(
                        args.output_dir,
                        f"profile_rank{dist.get_rank()}_{time.time()}.trace.json.gz",
                    )
                    print(f"End profile {output_path=}")
                    torch_profiler.stop()
                    torch_profiler.export_chrome_trace(output_path)

            # Training Step
            loss, metrics = run_forward(args, peagle_model, data, target_model)
            scaled_loss = loss / args.draft_accumulation_steps
            scaled_loss.backward()

            if global_step % args.draft_accumulation_steps == 0:
                optimizer.step()

            # Logging
            if global_step % (args.log_interval * args.draft_accumulation_steps) == 0:
                record_metrics(
                    args,
                    metrics,
                    global_step // args.draft_accumulation_steps,
                    tracker,
                    optimizer,
                    mode="train",
                )

            if dist.get_rank() == 0:
                time_per_step = time.time() - last_time
                last_time = time.time()
                acc = metrics.get("full_acc_sum", torch.tensor(0.0))
                acc_total = metrics.get("full_acc_total", torch.tensor(1.0))
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{(acc / acc_total.clamp_min(1)).item():.4f}",
                        "time": f"{time_per_step:.2f}s",
                    }
                )

            # Evaluation
            if (
                args.eval_data_path is not None
                and eval_dataloader is not None
                and global_step % (args.eval_interval * args.draft_accumulation_steps)
                == 0
            ):
                draft_model.eval()
                eval_metrics_accum = {}

                for eval_data in tqdm(
                    eval_dataloader, desc=f"Evaluating Epoch {epoch}"
                ):
                    with torch.no_grad():
                        _, eval_m = run_forward(
                            args, peagle_model, eval_data, target_model
                        )
                        for k, v in eval_m.items():
                            if k not in eval_metrics_accum:
                                eval_metrics_accum[k] = []
                            eval_metrics_accum[k].append(v)

                avg_metrics = {
                    k: torch.stack(v).mean() for k, v in eval_metrics_accum.items()
                }
                record_metrics(
                    args,
                    avg_metrics,
                    global_step // args.draft_accumulation_steps,
                    tracker,
                    mode="eval",
                )
                draft_model.train()

            # Save Checkpoints
            if global_step % args.save_interval == 0:
                save_checkpoints(args, epoch, global_step, peagle_model, optimizer)

            if args.max_num_steps is not None and global_step >= args.max_num_steps:
                break

        if args.max_num_steps is not None and global_step >= args.max_num_steps:
            break

    if global_step % args.save_interval != 0:
        print_on_rank0(
            f"Training completed at step {global_step}, saving final checkpoint..."
        )
        save_checkpoints(args, epoch, global_step, peagle_model, optimizer)

    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
