import argparse
import hashlib
import os
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from specforge import (
    AutoDistributedTargetModel,
    AutoDraftModelConfig,
    AutoEagle3DraftModel,
    Eagle3Model,
    generate_eagle3_targets,
)
from specforge.checkpoint.fsdp2 import load_checkpoint, save_checkpoint
from specforge.data import (
    build_eagle3_dataset,
    generate_vocab_mapping_file,
    prepare_dp_dataloaders,
)
from specforge.distributed import (
    destroy_distributed,
    get_device_mesh,
    get_dp_group,
    init_distributed,
)
from specforge.lr_scheduler import CosineAnnealingWarmupLR
from specforge.tracker import create_tracker, get_tracker_class
from specforge.utils import get_last_checkpoint, print_with_rank, rank_0_priority


def parse_args():
    parser = argparse.ArgumentParser(description="Train Eagle3 with online data")

    # add model-related arguments
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--draft-model-config", type=str, required=True)
    parser.add_argument(
        "--embedding-key",
        type=str,
        default="model.embed_tokens.weight",
        help="The key of the embedding weight to load from the target model",
    )

    # add training-related arguments
    parser.add_argument("--train-data-path", type=str, required=True)
    parser.add_argument("--eval-data-path", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-grad-norm", type=float, default=0.5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument(
        "--ttt-length",
        type=int,
        default=7,
        help="The length for Test-Time Training (TTT).",
    )

    # data processing type
    parser.add_argument("--chat-template", type=str, default="llama3")

    # distributed training
    parser.add_argument("--tp-size", type=int, default=1)

    # other args
    parser.add_argument("--cache-key", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=2000)
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
        choices=["wandb", "tensorboard", "swanlab", "none"],
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

    parser.add_argument("--build-dataset-num-proc", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-start-step", type=int, default=30)
    parser.add_argument("--profile-num-steps", type=int, default=4)
    parser.add_argument("--profile-record-shapes", action="store_true")

    args = parser.parse_args()

    return parser, args


def print_on_rank0(message):
    if dist.get_rank() == 0:
        print(message)


def parse_ckpt_info(checkpoint_path):
    # parse the global step from the checkpoint path
    path = Path(checkpoint_path)
    global_step = int(path.name.split("_global_step_")[-1])
    epoch = int(path.name.split("epoch_")[-1].split("_global_step_")[0])
    return global_step, epoch


@contextmanager
def accumulate_gradients(eagle3_model, gradient_accumulation_steps, global_step):
    """
    This context manager is used to accumulate gradients.

    Args:
        eagle3_model: The Eagle3 model.
        gradient_accumulation_steps: The number of steps to accumulate gradients.
        global_step: The current global step.

    Yields:
        None
    """
    global_step = global_step + 1
    if global_step % gradient_accumulation_steps == 0:
        eagle3_model.set_requires_gradient_sync(True)
    elif global_step % gradient_accumulation_steps == 1:
        eagle3_model.set_requires_gradient_sync(False)
    yield


def should_sync_gradients(global_step, gradient_accumulation_steps):
    global_step = global_step + 1
    return global_step % gradient_accumulation_steps == 0


def main():
    # initialize
    parser, args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank("Initialized distributed environment")

    tracker_class = get_tracker_class(args.report_to)
    if tracker_class:
        tracker_class.validate_args(parser, args)
    else:
        parser.error(f"Unknown tracker: {args.report_to}")

    tracker = create_tracker(args, args.output_dir)

    # detecting last ckpt for draft model
    draft_model_last_checkpoint = None
    if args.resume and os.path.isdir(args.output_dir):
        print_on_rank0(args.output_dir)
        draft_model_last_checkpoint = get_last_checkpoint(args.output_dir)
        print_on_rank0(f"Last checkpoint detected: {draft_model_last_checkpoint}")

    # build target and draft model
    if args.tp_size > 1:
        # to avoid CPU RAM OOM, we directly init the model on CUDA
        target_model = AutoDistributedTargetModel.from_pretrained(
            pretrained_model_name_or_path=args.target_model_path,
            torch_dtype=torch.bfloat16,
            cache_dir=args.cache_dir,
            device="cuda",
        ).eval()
    else:
        target_model = (
            AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=args.target_model_path,
                torch_dtype=torch.bfloat16,
                cache_dir=args.cache_dir,
            )
            .eval()
            .cuda()
        )
    print_with_rank("Initialized target model")
    # load model with resume
    draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)
    if draft_model_last_checkpoint:
        draft_model = (
            AutoEagle3DraftModel.from_pretrained(
                draft_model_last_checkpoint, attention_backend=args.attention_backend
            )
            .cuda()
            .to(torch.bfloat16)
        )
    else:
        draft_model = (
            AutoEagle3DraftModel.from_config(
                draft_model_config, attention_backend=args.attention_backend
            )
            .cuda()
            .to(torch.bfloat16)
        )
    draft_model.load_embedding(args.target_model_path, embedding_key=args.embedding_key)
    draft_model.freeze_embedding()
    print_with_rank("Initialized draft model")

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
        args.batch_size,
        num_workers=4,
        shuffle=True,
        process_group=get_dp_group(),
    )
    print_with_rank("Initialized train dataloader")

    # we load the vocab mapping then
    draft_model.load_vocab_mapping(vocab_mapping_path)
    print_with_rank("Loaded vocab mapping")

    if args.eval_data_path is not None:
        eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
        eval_eagle3_dataset = build_eagle3_dataset(
            eval_dataset,
            tokenizer,
            args.chat_template,
            args.max_length,
            num_proc=args.build_dataset_num_proc,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.batch_size,
            num_workers=4,
            shuffle=False,
            process_group=get_dp_group(),
        )
        print_with_rank("Initialized eval dataloader")

    # build Eagle3 model
    # broadcast draft model
    eagle3_model = Eagle3Model(
        draft_model=draft_model,
        length=args.ttt_length,
        attention_backend=args.attention_backend,
    )
    # eagle3_model = DDP(eagle3_model, find_unused_parameters=True)
    eagle3_model = fully_shard(
        eagle3_model,
        mesh=get_device_mesh(),
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
        ),
    )
    print_with_rank("Initialized Eagle3 FSDP model")

    # build other components
    optimizer = torch.optim.AdamW(
        eagle3_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    total_steps = args.num_epochs * len(train_dataloader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = CosineAnnealingWarmupLR(
        optimizer, total_steps=total_steps, warmup_steps=warmup_steps
    )
    print_with_rank("Initialized optimizer and scheduler")

    # global_step
    global_step = 0
    start_epoch = 0
    if draft_model_last_checkpoint is not None:
        print_on_rank0(
            f"Resuming draft model training from checkpoint: {draft_model_last_checkpoint}"
        )
        load_checkpoint(eagle3_model, optimizer, scheduler, draft_model_last_checkpoint)
        global_step, start_epoch = parse_ckpt_info(draft_model_last_checkpoint)
        print_on_rank0(f"Resuming from epoch {start_epoch}")

    dist.barrier()

    last_time = time.time()

    # start running
    print_on_rank0(f"Starting training from epoch {start_epoch}")
    for epoch in range(start_epoch, args.num_epochs):
        # Run training
        train_dataloader.sampler.set_epoch(epoch + 1)
        draft_model.train()
        epoch_acces = [[] for _ in range(eagle3_model.length)]
        epoch_plosses = [[] for _ in range(eagle3_model.length)]

        for batch_index, data in enumerate(
            tqdm(train_dataloader, desc=f"Training Epoch {epoch}")
        ):
            if args.profile and epoch == 0:
                if batch_index == args.profile_start_step:
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
                if batch_index == args.profile_start_step + args.profile_num_steps:
                    output_path = os.path.join(
                        os.environ["SGLANG_TORCH_PROFILER_DIR"],
                        f"debug_rank{torch.distributed.get_rank()}_{time.time()}.trace.json.gz",
                    )
                    print(f"End profile {output_path=}")
                    torch_profiler.stop()
                    torch_profiler.export_chrome_trace(output_path)

            optimizer.zero_grad()

            input_ids = data["input_ids"].cuda()
            attention_mask = data["attention_mask"].cuda()
            loss_mask = data["loss_mask"].cuda()

            hidden_states, target, loss_mask, input_ids = generate_eagle3_targets(
                target_model=target_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
            )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                plosses, _, acces = eagle3_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    loss_mask=loss_mask,
                    hidden_states=hidden_states,
                    target=target,
                )

            with accumulate_gradients(
                eagle3_model, args.gradient_accumulation_steps, global_step
            ):
                # calculate weighted loss
                ploss_weight = [0.8**i for i in range(len(plosses))]
                ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
                ploss.backward()

            global_step += 1

            if should_sync_gradients(global_step, args.gradient_accumulation_steps):
                torch.nn.utils.clip_grad_norm_(
                    eagle3_model.parameters(), args.clip_grad_norm
                )
                optimizer.step()
                scheduler.step()

                logdict = {"train/lr": optimizer.param_groups[0]["lr"]}
                for i in range(len(plosses)):
                    logdict[f"train/ploss_{i}"] = plosses[i].item()
                for i in range(len(acces)):
                    logdict[f"train/acc_{i}"] = acces[i]
                tracker.log(logdict, step=global_step)

                epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
                epoch_plosses = [
                    epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))
                ]

                if args.verbose:
                    print(
                        f"[{dist.get_rank()}] time={(time.time() - last_time):.3}s shape={data['input_ids'].shape}"
                    )
                    last_time = time.time()

            if global_step % args.save_interval == 0:
                # Save the model
                epoch_output_dir = os.path.join(
                    args.output_dir, f"epoch_{epoch}_global_step_{global_step}"
                )
                if dist.get_rank() == 0:
                    os.makedirs(epoch_output_dir, exist_ok=True)
                dist.barrier()
                save_checkpoint(eagle3_model, optimizer, scheduler, epoch_output_dir)

        epoch_logdict = {}
        for i in range(len(epoch_acces)):
            acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
            dist.all_reduce(acc_i)
            acc_i = (acc_i / dist.get_world_size()).item()
            epoch_logdict[f"train/epoch_acc_{i}"] = acc_i
            print_on_rank0(
                f"Train Epoch [{epoch + 1}/{args.num_epochs}], position {i},  Acc: {acc_i:.2f}"
            )

        for i in range(len(epoch_plosses)):
            loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
            dist.all_reduce(loss_i)
            loss_i = (loss_i / dist.get_world_size()).item()
            epoch_logdict[f"train/epoch_ploss_{i}"] = loss_i
            print_on_rank0(
                f"Train Epoch [{epoch + 1}/{args.num_epochs}], position {i}, pLoss: {loss_i:.2f}"
            )
        tracker.log(epoch_logdict, step=global_step)

        # run evaluation
        if args.eval_data_path is not None and epoch % args.eval_interval == 0:
            # Run evaluation
            draft_model.eval()
            eval_acces = [[] for _ in range(eagle3_model.length)]
            eval_plosses = [[] for _ in range(eagle3_model.length)]

            for data in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch}"):
                input_ids = data["input_ids"].cuda()
                attention_mask = data["attention_mask"].cuda()
                loss_mask = data["loss_mask"].cuda()

                hidden_states, target, loss_mask, input_ids = generate_eagle3_targets(
                    target_model=target_model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    loss_mask=loss_mask,
                )

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    plosses, _, acces = eagle3_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        loss_mask=loss_mask,
                        hidden_states=hidden_states,
                        target=target,
                    )

                eval_acces = [eval_acces[i] + [acces[i]] for i in range(len(acces))]
                eval_plosses = [
                    eval_plosses[i] + [plosses[i].item()] for i in range(len(plosses))
                ]

            # Log epoch-level evaluation metrics
            eval_logdict = {}
            for i in range(len(eval_acces)):
                acc_i = torch.tensor(eval_acces[i]).cuda().mean()
                dist.all_reduce(acc_i)
                acc_i = (acc_i / dist.get_world_size()).item()
                eval_logdict[f"eval/epoch_acc_{i}"] = acc_i
                print_on_rank0(
                    f"Eval Epoch [{epoch + 1}/{args.num_epochs}], position {i},  Acc: {acc_i:.2f}"
                )

            for i in range(len(eval_plosses)):
                loss_i = torch.tensor(eval_plosses[i]).cuda().mean()
                dist.all_reduce(loss_i)
                loss_i = (loss_i / dist.get_world_size()).item()
                eval_logdict[f"eval/epoch_ploss_{i}"] = loss_i
                print_on_rank0(
                    f"Eval Epoch [{epoch + 1}/{args.num_epochs}], position {i}, pLoss: {loss_i:.2f}"
                )
            tracker.log(eval_logdict, step=global_step)

    # Close the tracker
    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
