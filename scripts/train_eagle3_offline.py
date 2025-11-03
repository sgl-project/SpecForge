import argparse
import hashlib
import math
import os
import time
from collections import defaultdict

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from tqdm import tqdm
from transformers import AutoTokenizer

from specforge import AutoDraftModelConfig, AutoEagle3DraftModel, OfflineEagle3Model
from specforge.data import (
    build_eagle3_dataset,
    build_offline_eagle3_dataset,
    generate_vocab_mapping_file,
    prepare_dp_dataloaders,
)
from specforge.distributed import (
    destroy_distributed,
    get_dp_device_mesh,
    get_dp_group,
    init_distributed,
)
from specforge.modeling.target.target_head import TargetHead
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker, get_tracker_class
from specforge.utils import (
    create_draft_config_from_target,
    get_full_optimizer_state,
    get_last_checkpoint,
    print_on_rank0,
    print_with_rank,
    rank_0_priority,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Eagle3 with offline data")

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
        "--lm-head-key",
        type=str,
        default="lm_head.weight",
        help="The key of the lm head weight to load from the target model",
    )

    # add training-related arguments
    parser.add_argument("--train-data-path", type=str, required=True)
    parser.add_argument("--train-hidden-states-path", type=str, required=True)
    parser.add_argument("--eval-data-path", type=str, default=None)
    parser.add_argument("--eval-hidden-states-path", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--draft-global-batch-size", type=int, default=16)
    parser.add_argument("--draft-micro-batch-size", type=int, default=1)
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
        "--log-steps", type=int, default=50, help="Log training metrics every N steps"
    )
    parser.add_argument(
        "--ttt-length",
        type=int,
        default=7,
        help="The length for Test-Time Training (TTT).",
    )
    parser.add_argument("--draft-attention-backend", type=str, default="flex_attention")
    # data processing type
    parser.add_argument("--chat-template", type=str, default="llama3")
    parser.add_argument(
        "--is-preformatted",
        action="store_true",
        help="Whether the input data is preformatted text with the chat template already applied to the conversation messages.",
    )

    # distributed training
    parser.add_argument("--tp-size", type=int, default=1)

    # other args
    parser.add_argument("--cache-key", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dist-timeout",
        type=int,
        default=20,
        help="Timeout for collective communication in minutes",
    )
    # resume
    parser.add_argument("--resume", action="store_true")

    # report backend
    parser.add_argument(
        "--report-to",
        type=str,
        default="none",
        choices=["wandb", "tensorboard", "swanlab", "mlflow", "none"],
        help="The integration to report results and logs to.",
    )
    # wandb-specific args
    parser.add_argument(
        "--wandb-project", type=str, default=None, help="The project name for W&B."
    )
    parser.add_argument(
        "--wandb-name", type=str, default=None, help="The run name for W&B."
    )
    parser.add_argument("--wandb-key", type=str, default=None, help="W&B API key.")
    # add swanlab-specific args ---
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

    parser.add_argument("--build-dataset-num-proc", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-start-step", type=int, default=30)
    parser.add_argument("--profile-num-steps", type=int, default=4)
    parser.add_argument("--profile-record-shapes", action="store_true")

    args = parser.parse_args()

    return parser, args


def main():
    # initialize
    parser, args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank("Initialized distributed environment")
    args.dp_size = dist.get_world_size() // args.tp_size
    args.draft_accumulation_steps = (
        args.draft_global_batch_size // args.dp_size // args.draft_micro_batch_size
    )
    assert (
        args.draft_accumulation_steps * args.draft_micro_batch_size * args.dp_size
        == args.draft_global_batch_size
    ), f"draft_global_batch_size={args.draft_global_batch_size} must be divisible by dp_size={args.dp_size} and micro_batch_size={args.draft_micro_batch_size}"
    print_with_rank(
        f"draft_accumulation_steps={args.draft_global_batch_size} // {args.dp_size} // {args.draft_micro_batch_size}={args.draft_accumulation_steps}"
    )

    # Validate report backend arguments
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
    target_head = TargetHead(args.target_model_path)
    target_head.load_weights(
        model_path=args.target_model_path,
        lm_head_key=args.lm_head_key,
        cache_dir=args.cache_dir,
    )
    target_head.freeze_weights()
    target_head = target_head.eval().cuda().to(torch.bfloat16)
    print_with_rank("Initialized target head")

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

    if draft_model_last_checkpoint:
        draft_model = (
            AutoEagle3DraftModel.from_pretrained(
                draft_model_last_checkpoint,
                attention_backend=args.draft_attention_backend,
            )
            .cuda()
            .to(torch.bfloat16)
        )
    else:
        draft_model = (
            AutoEagle3DraftModel.from_config(
                draft_model_config, attention_backend=args.draft_attention_backend
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
        train_eagle3_dataset_tmp = build_eagle3_dataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            is_preformatted=args.is_preformatted,
            max_length=args.max_length,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=cache_key,
            num_proc=args.build_dataset_num_proc,
        )
        vocab_mapping_path = generate_vocab_mapping_file(
            dataset=train_eagle3_dataset_tmp,
            target_vocab_size=draft_model_config.vocab_size,
            draft_vocab_size=draft_model_config.draft_vocab_size,
            cache_dir=os.path.join(args.cache_dir, "vocab_mapping"),
            cache_key=cache_key,
        )
        train_eagle3_dataset = build_offline_eagle3_dataset(
            args.train_hidden_states_path,
            args.max_length,
        )

    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.draft_micro_batch_size,
        num_workers=4,
        shuffle=True,
        process_group=get_dp_group(),
        pin_memory=True,
    )
    print_with_rank("Initialized train dataloader")

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

    # we load the vocab mapping then
    draft_model.load_vocab_mapping(vocab_mapping_path)
    print_with_rank("Loaded vocab mapping")

    if args.eval_data_path is not None:
        eval_eagle3_dataset = build_offline_eagle3_dataset(
            args.eval_hidden_states_path,
            args.max_length,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.draft_micro_batch_size,
            num_workers=4,
            shuffle=False,
            process_group=get_dp_group(),
            pin_memory=True,
        )
        print_with_rank("Initialized eval dataloader")

    # build Eagle3 model
    eagle3_model = OfflineEagle3Model(
        target_head=target_head,
        draft_model=draft_model,
        length=args.ttt_length,
        attention_backend=args.draft_attention_backend,
    )
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    fsdp_config = {"mesh": get_dp_device_mesh(), "mp_policy": mp_policy}
    fully_shard(eagle3_model, **fsdp_config)

    print_with_rank(f"Initialized Eagle3 FSDP model")
    global_step, batch_index = 0, 0
    log_dict = defaultdict(float)
    # build other components
    optimizer = BF16Optimizer(
        eagle3_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=args.total_steps,
    )
    print_with_rank("Initialized optimizer and scheduler")

    last_time = time.time()

    # start running
    for epoch in range(args.num_epochs):
        # Run training
        train_dataloader.sampler.set_epoch(epoch + 1)
        draft_model.train()
        epoch_acces = [[] for _ in range(eagle3_model.length)]
        epoch_plosses = [[] for _ in range(eagle3_model.length)]

        if dist.get_rank() == 0:
            progress_bar = tqdm(
                train_dataloader, desc=f"Training Epoch {epoch}", leave=True
            )
        else:
            progress_bar = train_dataloader

        for data in progress_bar:
            batch_index += 1
            if args.profile:
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

            # if batch_index % args.draft_accumulation_steps == 0:
            #     optimizer.zero_grad()
            plosses, _, acces = eagle3_model(
                input_ids=data["input_ids"].cuda(),  # [B, S]
                attention_mask=data["attention_mask"].cuda(),  # [B, S]
                loss_mask=data["loss_mask"]
                .unsqueeze(-1)
                .cuda(),  # [B, S, 1] This is different from the online version
                hidden_states=data["hidden_state"].cuda(),  # [B, S, D]
                target=data["target"].cuda(),  # [B, S, D*3]
            )
            acces = torch.stack(acces).cpu().tolist()

            # calculate weighted loss
            ploss_weight = [0.8**i for i in range(len(plosses))]
            ploss = (
                sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
                / args.draft_accumulation_steps
            )
            ploss.backward()
            log_dict["train/lr"] = optimizer.get_learning_rate()
            for i in range(len(plosses)):
                log_dict[f"train/ploss_{i}"] += (
                    plosses[i].item() / args.draft_accumulation_steps
                )
            for i in range(len(acces)):
                log_dict[f"train/acc_{i}"] += acces[i] / args.draft_accumulation_steps
            if batch_index % args.draft_accumulation_steps == 0:
                optimizer.step()
                global_step += 1
                # Pass global_step to the tracker
                if global_step % args.log_steps == 0:
                    tracker.log(log_dict, step=global_step)
                log_dict = defaultdict(float)

            epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
            epoch_plosses = [
                epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))
            ]

            if args.verbose:
                print(
                    f"[{dist.get_rank()}] time={(time.time() - last_time):.3}s shape={data['input_ids'].shape}"
                )
                last_time = time.time()

            if dist.get_rank() == 0:
                avg_loss = sum(pl.item() for pl in plosses) / len(plosses)
                avg_acc = sum(acces) / len(acces)
                progress_bar.set_postfix(
                    {"loss": f"{avg_loss:.2f}", "acc": f"{avg_acc:.2f}"}
                )

        # Log epoch-level training metrics
        train_epoch_logdict = {}
        for i in range(len(epoch_acces)):
            acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
            dist.all_reduce(acc_i)
            acc_i = (acc_i / dist.get_world_size()).item()
            train_epoch_logdict[f"train/epoch_acc_{i}"] = acc_i
            print_on_rank0(
                f"Train Epoch [{epoch + 1}/{args.num_epochs}], position {i},  Acc: {acc_i:.2f}"
            )
        for i in range(len(epoch_plosses)):
            loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
            dist.all_reduce(loss_i)
            loss_i = (loss_i / dist.get_world_size()).item()
            train_epoch_logdict[f"train/epoch_ploss_{i}"] = loss_i
            print_on_rank0(
                f"Train Epoch [{epoch + 1}/{args.num_epochs}], position {i}, pLoss: {loss_i:.2f}"
            )
        tracker.log(train_epoch_logdict, step=global_step)

        # run evaluation
        if args.eval_data_path is not None and epoch % args.eval_interval == 0:
            # Run evaluation
            draft_model.eval()
            eval_acces = [[] for _ in range(eagle3_model.length)]
            eval_plosses = [[] for _ in range(eagle3_model.length)]

            for data in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch}"):
                with torch.no_grad():
                    plosses, _, acces = eagle3_model(
                        input_ids=data["input_ids"].cuda(),
                        attention_mask=data["attention_mask"].cuda(),
                        loss_mask=data["loss_mask"].unsqueeze(-1).cuda(),
                        hidden_states=data["hidden_state"].cuda(),
                        target=data["target"].cuda(),
                    )
                acces = torch.stack(acces).cpu().tolist()
                eval_acces = [eval_acces[i] + [acces[i]] for i in range(len(acces))]
                eval_plosses = [
                    eval_plosses[i] + [plosses[i].item()] for i in range(len(plosses))
                ]

            # Log epoch-level evaluation metrics
            eval_epoch_logdict = {}
            for i in range(len(eval_acces)):
                acc_i = torch.tensor(eval_acces[i]).cuda().mean()
                dist.all_reduce(acc_i)
                acc_i = (acc_i / dist.get_world_size()).item()
                eval_epoch_logdict[f"eval/epoch_acc_{i}"] = acc_i
                print_on_rank0(
                    f"Eval Epoch [{epoch + 1}/{args.num_epochs}], position {i},  Acc: {acc_i:.2f}"
                )

            for i in range(len(epoch_plosses)):
                loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
                dist.all_reduce(loss_i)
                loss_i = (loss_i / dist.get_world_size()).item()
                eval_epoch_logdict[f"eval/epoch_ploss_{i}"] = loss_i
                print_on_rank0(
                    f"Eval Epoch [{epoch + 1}/{args.num_epochs}], position {i}, pLoss: {loss_i:.2f}"
                )
            tracker.log(eval_epoch_logdict, step=global_step)

        if epoch % args.save_interval == 0:
            # Save the model
            epoch_output_dir = os.path.join(args.output_dir, f"epoch_{epoch}")

            if dist.get_rank() == 0:
                os.makedirs(epoch_output_dir, exist_ok=True)
            dist.barrier()

            model_state_dict = eagle3_model.state_dict()

            state_to_save = {
                "epoch": epoch,
                "args": args,
            }

            optimizer_state_dict = optimizer.state_dict()
            optimizer_state_dict["optimizer_state_dict"] = get_full_optimizer_state(
                optimizer_state_dict["optimizer_state_dict"]
            )

            state_to_save.update(optimizer_state_dict)

            draft_model_state_dict = {
                k.replace("draft_model.", ""): (
                    v.full_tensor()
                    if isinstance(v, torch.distributed.tensor.DTensor)
                    else v
                )
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
                draft_model.save_pretrained(
                    epoch_output_dir,
                    state_dict=draft_model_state_dict,
                )
                print_on_rank0(f"Saved model configuration to {epoch_output_dir}")
            dist.barrier()

    # Close the tracker at the end of training
    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
