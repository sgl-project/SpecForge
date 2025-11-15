import argparse
import hashlib
import math
import os
import time
from argparse import ArgumentParser, Namespace
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from specforge import AutoDraftModelConfig, AutoEagle3DraftModel, OnlineEagle3Model
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
from specforge.modeling.target import Eagle3TargetModel, get_eagle3_target_model
from specforge.optimizer import BF16Optimizer
from specforge.tracker import Tracker, create_tracker, get_tracker_class
from specforge.utils import (
    create_draft_config_from_target,
    get_last_checkpoint,
    print_on_rank0,
    print_with_rank,
    rank_0_priority,
)


def parse_args() -> Tuple[ArgumentParser, Namespace]:
    """
    This function is used to parse the arguments for the training script.
    """
    parser = argparse.ArgumentParser(description="Train Eagle3 with online data")

    # add model-related arguments
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument(
        "--draft-model-config",
        type=str,
        required=False,
        help="Draft model config path. If not provided, will auto-generate from target model.",
    )
    parser.add_argument(
        "--embedding-key",
        type=str,
        default="model.embed_tokens.weight",
        help="The key of the embedding weight to load from the target model",
    )
    parser.add_argument(
        "--is-vlm", action="store_true", help="Whether the target model is a VLM"
    )

    # add training-related arguments
    parser.add_argument("--train-data-path", type=str, required=True)
    parser.add_argument("--eval-data-path", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--warmup-ratio", type=float, default=0.015)
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Total training steps. If not provided, will be calculated as num_epochs * steps_per_epoch",
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Log training metrics every N steps",
    )
    parser.add_argument(
        "--ttt-length",
        type=int,
        default=7,
        help="The length for Test-Time Training (TTT).",
    )

    # data processing type
    parser.add_argument("--chat-template", type=str, default="llama3")
    parser.add_argument(
        "--is-preformatted",
        action="store_true",
        help="Whether the input data is preformatted text with the chat template already applied to the conversation messages.",
    )

    # distributed training
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--draft-accumulation-steps", type=int, default=1)

    # other args
    parser.add_argument("--cache-key", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dist-timeout",
        type=int,
        default=20,
        help="Timeout for collective communication in minutes",
    )
    parser.add_argument("--attention-backend", type=str, default="flex_attention")

    # resume
    parser.add_argument("--resume", action="store_true")

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

    # vlm related args
    parser.add_argument(
        "--min-pixels", type=int, default=50176
    )  # 64*28*28 for qwen2.5-vl
    parser.add_argument(
        "--max-pixels", type=int, default=802816
    )  # 1024*28*28 for qwen2.5-vl

    parser.add_argument("--build-dataset-num-proc", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-start-step", type=int, default=30)
    parser.add_argument("--profile-num-steps", type=int, default=4)
    parser.add_argument("--profile-record-shapes", action="store_true")
    parser.add_argument(
        "--target-model-backend",
        type=str,
        default="sglang",
        choices=["sglang", "hf", "custom"],
        help="The backend of the target model",
    )

    args = parser.parse_args()
    return parser, args


def build_tracker(args: Namespace, parser: ArgumentParser) -> Tracker:
    """
    Build the experiment tracker according to the report_to argument.

    Args:
        args: The arguments for the training script.
        parser: The parser for the training script.

    Returns:
        The experiment tracker.
    """
    tracker_class = get_tracker_class(args.report_to)
    if tracker_class:
        tracker_class.validate_args(parser, args)
    else:
        parser.error(f"Unknown tracker: {args.report_to}")
    tracker = create_tracker(args, args.output_dir)
    return tracker


def build_target_model(
    args: Namespace, draft_model_config: AutoDraftModelConfig
) -> Eagle3TargetModel:
    """
    Build the target model according to the arguments.

    Args:
        args: The arguments for the training script.
        draft_model_config: The draft model config.

    Returns:
        The target model.
    """
    target_model = get_eagle3_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device="cuda",
        cache_dir=args.cache_dir,
    )

    # set the aux hidden states layers
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

    if args.is_vlm:
        processor = AutoProcessor.from_pretrained(
            args.target_model_path,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
    else:
        processor = None

    return target_model, processor


def sanity_check(args: Namespace) -> None:
    """
    Perform sanity checks on the arguments.

    Args:
        args: The arguments for the training script.

    Returns:
        None
    """
    args.dp_size = dist.get_world_size() // args.tp_size
    args.target_batch_size = args.tp_size * args.batch_size


def build_draft_model(args: Namespace) -> Tuple[AutoDraftModelConfig, nn.Module]:
    # Handle draft model config
    if args.draft_model_config is None:
        # Auto-generate and save config file
        auto_config_path = create_draft_config_from_target(
            target_model_path=args.target_model_path, cache_dir=args.cache_dir
        )
        draft_model_config = AutoDraftModelConfig.from_file(auto_config_path)
    else:
        # Use provided config file
        draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)

    # detecting last ckpt for draft model
    draft_model_last_checkpoint = None
    if args.resume and os.path.isdir(args.output_dir):
        print_on_rank0(args.output_dir)
        draft_model_last_checkpoint = get_last_checkpoint(args.output_dir)
        print_on_rank0(f"Last checkpoint detected: {draft_model_last_checkpoint}")

    if draft_model_last_checkpoint:
        draft_model = AutoEagle3DraftModel.from_pretrained(
            draft_model_last_checkpoint,
            attention_backend=args.attention_backend,
            torch_dtype=torch.bfloat16,
        ).cuda()
    else:
        draft_model = AutoEagle3DraftModel.from_config(
            draft_model_config,
            attention_backend=args.attention_backend,
            torch_dtype=torch.bfloat16,
        ).cuda()

    draft_model.load_embedding(args.target_model_path, embedding_key=args.embedding_key)
    draft_model.freeze_embedding()
    return draft_model_config, draft_model


def build_dataloaders(
    args: Namespace,
    draft_model_config: AutoDraftModelConfig,
    processor: Optional[AutoProcessor] = None,
) -> Tuple[DataLoader, str, Optional[DataLoader]]:
    # build dataloaders
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    # convert to dataloader
    cache_params_string = (
        f"{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"  # Tokenizer may also different
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    with rank_0_priority():
        train_eagle3_dataset = build_eagle3_dataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=cache_key,
            is_vlm=args.is_vlm,
            is_preformatted=args.is_preformatted,
            processor=processor,
            num_proc=args.build_dataset_num_proc,
        )
        vocab_mapping_path = generate_vocab_mapping_file(
            dataset=train_eagle3_dataset,
            target_vocab_size=draft_model_config.vocab_size,
            draft_vocab_size=draft_model_config.draft_vocab_size,
            cache_dir=os.path.join(args.cache_dir, "vocab_mapping"),
            cache_key=cache_key,
        )
    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.target_batch_size,
        num_workers=4,
        shuffle=True,
        process_group=get_dp_group(),
        is_vlm=args.is_vlm,
    )

    if args.eval_data_path is not None:
        eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
        eval_eagle3_dataset = build_eagle3_dataset(
            eval_dataset,
            tokenizer,
            args.chat_template,
            args.max_length,
            is_vlm=args.is_vlm,
            processor=processor,
            num_proc=args.build_dataset_num_proc,
            is_preformatted=args.is_preformatted,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.target_batch_size,
            num_workers=4,
            shuffle=False,
            process_group=get_dp_group(),
            is_vlm=args.is_vlm,
        )
        print_with_rank("Initialized eval dataloader")
    else:
        eval_dataloader = None
    return (
        train_dataloader,
        vocab_mapping_path,
        eval_dataloader,
    )


def save_checkpoints(
    args: Namespace,
    epoch: int,
    step: int,
    eagle3_model: nn.Module,
    optimizer: Optimizer,
):
    epoch_output_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(epoch_output_dir, exist_ok=True)
    dist.barrier()

    with FSDP.state_dict_type(eagle3_model, StateDictType.FULL_STATE_DICT):
        model_state_dict = eagle3_model.state_dict()
        state_to_save = {
            "epoch": epoch,
            "global_step": step,
            "args": args,
        }
        state_to_save.update(optimizer.state_dict())
        draft_model_state_dict = {
            k.replace("draft_model.", ""): v
            for k, v in model_state_dict.items()
            if "draft_model." in k and "embed" not in k.lower()
        }

        if dist.get_rank() == 0:
            torch.save(
                state_to_save,
                os.path.join(epoch_output_dir, "training_state.pt"),
            )
            print_on_rank0(
                f"Saved full training state to {epoch_output_dir}/training_state.pt"
            )
            eagle3_model.draft_model.save_pretrained(
                epoch_output_dir,
                state_dict=draft_model_state_dict,
            )
            print_on_rank0(f"Saved model configuration to {epoch_output_dir}")
        dist.barrier()


def run_forward(
    args: Namespace,
    eagle3_model: nn.Module,
    data: dict,
    target_model: Optional[Eagle3TargetModel] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    kwargs = {
        "input_ids": data["input_ids"].cuda(),
        "attention_mask": data["attention_mask"].cuda(),
        "loss_mask": data["loss_mask"].cuda(),
        "is_vlm": args.is_vlm,
    }
    if args.is_vlm:
        if "pixel_values" in data:
            kwargs["pixel_values"] = data["pixel_values"].cuda()
        if "image_grid_thw" in data:
            kwargs["image_grid_thw"] = data["image_grid_thw"].cuda()
    eagle3_data = target_model.generate_eagle3_data(**kwargs)
    eagle3_data.input_ids = get_dp_data_shard_from_tp(eagle3_data.input_ids)
    eagle3_data.attention_mask = get_dp_data_shard_from_tp(eagle3_data.attention_mask)
    eagle3_data.loss_mask = get_dp_data_shard_from_tp(eagle3_data.loss_mask)
    eagle3_data.target = get_dp_data_shard_from_tp(eagle3_data.target)
    eagle3_data.hidden_states = get_dp_data_shard_from_tp(eagle3_data.hidden_states)

    plosses, _, acces = eagle3_model(
        input_ids=eagle3_data.input_ids,
        attention_mask=eagle3_data.attention_mask,
        loss_mask=eagle3_data.loss_mask,
        target=eagle3_data.target,
        hidden_states=eagle3_data.hidden_states,
    )
    return plosses, acces


def run_backward_and_update(
    args: Namespace, plosses: List[torch.Tensor], optimizer: Optimizer, global_step: int
) -> None:
    ploss_weight = [0.8**i for i in range(len(plosses))]
    ploss = (
        sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
        / args.draft_accumulation_steps
    )
    ploss.backward()

    if global_step % args.draft_accumulation_steps == 0:
        optimizer.step()


def record_metrcs(
    args: Namespace,
    accuracies: List[torch.Tensor],
    plosses: List[torch.Tensor],
    global_step: int,
    tracker: Tracker,
    optimizer: Optional[Optimizer] = None,
    mode: str = "train",
) -> None:
    logdict = {}

    if mode == "train" and optimizer is not None:
        logdict["train/lr"] = optimizer.get_learning_rate()

    accuracies = torch.stack(accuracies)
    plosses = torch.stack(plosses)

    assert accuracies.shape[0] == args.ttt_length
    dist.all_reduce(accuracies, op=dist.ReduceOp.AVG)
    accuracies = accuracies.cpu().tolist()
    for i in range(len(accuracies)):
        logdict[f"{mode}/acc_{i}"] = accuracies[i]
        print_on_rank0(
            f"Eval - Step {global_step} [{global_step + 1}/{args.num_epochs}], position {i},  Acc: {accuracies[i]:.2f}"
        )

    dist.all_reduce(plosses, op=dist.ReduceOp.AVG)
    plosses = plosses.cpu().tolist()
    for i in range(len(plosses)):
        logdict[f"{mode}/ploss_{i}"] = plosses[i]
        print_on_rank0(
            f"Eval - Step {global_step} [{global_step + 1}/{args.num_epochs}], position {i}, pLoss: {plosses[i]}"
        )
    tracker.log(logdict, step=global_step)


def get_dp_data_shard_from_tp(tensor: torch.Tensor) -> torch.Tensor:
    """
    Get the data shard from the tensor.
    """
    tp_size = dist.get_world_size(get_tp_group())
    tp_rank = dist.get_rank(get_tp_group())
    return tensor.chunk(tp_size, dim=0)[tp_rank]


def main():
    # ================================================
    # 1. Initialize
    # ================================================
    parser, args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    sanity_check(args)
    print_with_rank("Initialized distributed environment")

    # ================================================
    # 2. Build models
    # ================================================
    draft_model_config, draft_model = build_draft_model(args)
    target_model, processor = build_target_model(args, draft_model_config)

    # ================================================
    # 3. Build dataloader
    # ================================================
    train_dataloader, vocab_mapping_path, eval_dataloader = build_dataloaders(
        args, draft_model_config, processor
    )

    # we load the vocab mapping then
    draft_model.load_vocab_mapping(vocab_mapping_path)
    print_with_rank("Loaded vocab mapping")

    # Calculate total steps if not provided
    if args.total_steps is None:
        steps_per_epoch = math.ceil(
            len(train_dataloader) / args.draft_accumulation_steps
        )
        args.total_steps = args.num_epochs * steps_per_epoch
        print_with_rank(
            f"Auto-calculated total_steps: {args.total_steps} (num_epochs={args.num_epochs} * steps_per_epoch={steps_per_epoch})"
        )
    else:
        print_with_rank(f"Using provided total_steps: {args.total_steps}")

    # ================================================
    # 4. Build Eagle3 model
    # ================================================

    eagle3_model = OnlineEagle3Model(
        draft_model=draft_model,
        length=args.ttt_length,
        attention_backend=args.attention_backend,
    )

    eagle3_model = FSDP(
        eagle3_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        process_group=dist.group.WORLD,  # the draft model should run dp for all processes
    )
    print_with_rank("Initialized Eagle3 FSDP model")

    # ================================================
    # 5. Build optimizer and scheduler
    # ================================================
    optimizer = BF16Optimizer(
        draft_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=args.total_steps,
    )
    print_with_rank("Initialized optimizer and scheduler")

    # ================================================
    # 6. Build tracker
    # ================================================
    tracker = build_tracker(args, parser)
    global_step = 0
    start_epoch = 0
    dist.barrier()

    last_time = time.time()

    # ================================================
    # 7. Start training
    # ================================================
    print_on_rank0(f"Starting training from epoch {start_epoch}")

    for epoch in range(start_epoch, args.num_epochs):
        # Run training
        train_dataloader.sampler.set_epoch(epoch + 1)
        draft_model.train()

        if dist.get_rank() == 0:
            progress_bar = tqdm(
                train_dataloader, desc=f"Training Epoch {epoch}", leave=True
            )
        else:
            progress_bar = train_dataloader

        for data in progress_bar:
            global_step += 1

            # ================================================
            # 7.1 Training Step
            # ================================================
            plosses, acces = run_forward(args, eagle3_model, data, target_model)
            run_backward_and_update(args, plosses, optimizer, global_step)

            # log training metrics
            if global_step % args.log_interval == 0:
                record_metrcs(
                    args, acces, plosses, global_step, tracker, optimizer, mode="train"
                )

            if dist.get_rank() == 0:
                time_per_step = time.time() - last_time
                last_time = time.time()
                avg_loss = sum(pl for pl in plosses) / len(plosses)
                avg_acc = sum(acces) / len(acces)
                progress_bar.set_postfix(
                    {
                        "loss": f"{avg_loss:.2f}",
                        "acc": f"{avg_acc:.2f}",
                        "time": f"{time_per_step:.2f}s",
                    }
                )

            # ================================================
            # 7.2 Evaluation Step
            # ================================================
            if (
                args.eval_data_path is not None
                and global_step % args.eval_interval == 0
            ):
                # Run evaluation
                draft_model.eval()
                eval_acces = [[] for _ in range(eagle3_model.length)]
                eval_plosses = [[] for _ in range(eagle3_model.length)]

                for data in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch}"):
                    with torch.no_grad():
                        plosses, acces = run_forward(
                            args, eagle3_model, data, target_model
                        )
                        eval_acces = [
                            eval_acces[i] + [acces[i]] for i in range(len(acces))
                        ]
                        eval_plosses = [
                            eval_plosses[i] + [plosses[i]] for i in range(len(plosses))
                        ]

                # compute average over all minibatches
                eval_acces = [torch.stack(acc).mean() for acc in eval_acces]
                eval_plosses = [torch.stack(pl).mean() for pl in eval_plosses]

                record_metrcs(
                    args,
                    eval_acces,
                    eval_plosses,
                    global_step,
                    tracker,
                    mode="eval",
                )

            # ================================================
            # 7.3 Save Checkpoints
            # ================================================
            if global_step % args.save_interval == 0:
                # Save the model
                save_checkpoints(args, epoch, global_step, eagle3_model, optimizer)

    # Close the tracker
    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
