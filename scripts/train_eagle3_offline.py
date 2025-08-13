import argparse
import hashlib
import os

import torch
import torch.distributed as dist
import wandb
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from tqdm import tqdm
from transformers import AutoTokenizer

from specforge import AutoDraftModelConfig, AutoEagle3DraftModel, OfflineEagle3Model
from specforge.data import (
    build_eagle3_dataset,
    build_offline_eagle3_dataset,
    generate_vocab_mapping_file,
    prepare_dp_dataloaders,
)
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.lr_scheduler import CosineAnnealingWarmupLR
from specforge.modeling.target.target_head import TargetHead
from specforge.utils import print_with_rank, rank_0_priority, validate_wandb_args


def parse_args():
    parser = argparse.ArgumentParser(description="Train Eagle3 with offline data")

    # add model-related arguments
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--draft-model-config", type=str, required=True)
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
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
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dist-timeout",
        type=int,
        default=20,
        help="Timeout for collective communication in minutes",
    )

    # wandb wandb args
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-key", type=str, default=None)

    args = parser.parse_args()

    return parser, args


def init_wandb(args):
    wandb.login(key=args.wandb_key)
    wandb.init(project=args.wandb_project, name=args.wandb_name)


def wandb_log_if_initialized(log_dict):
    if dist.get_rank() == 0 and wandb.run is not None:
        wandb.log(log_dict)


def print_on_rank0(message):
    if dist.get_rank() == 0:
        print(message)


def main():
    # initialize
    parser, args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank(f"Initialized distributed environment")

    # Validate wandb arguments
    validate_wandb_args(parser, args)

    if args.wandb and dist.get_rank() == 0:
        init_wandb(args)

    # build target and draft model
    target_head = TargetHead(args.target_model_path)
    target_head.load_weights(
        model_path=args.target_model_path, lm_head_key=args.lm_head_key
    )
    target_head.freeze_weights()
    target_head = target_head.eval().cuda().to(torch.bfloat16)
    print_with_rank(f"Initialized target head")

    draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)
    draft_model = (
        AutoEagle3DraftModel.from_config(draft_model_config).cuda().to(torch.bfloat16)
    )
    draft_model.load_embedding(args.target_model_path, embedding_key=args.embedding_key)
    draft_model.freeze_embedding()
    print_with_rank(f"Initialized draft model")

    # build dataloaders
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    # convert to dataloader
    cache_key = hashlib.md5(args.train_data_path.encode()).hexdigest()
    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    with rank_0_priority():
        train_eagle3_dataset_tmp = build_eagle3_dataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=cache_key,
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
        args.batch_size,
        num_workers=4,
        shuffle=True,
        process_group=get_dp_group(),
    )
    print_with_rank(f"Initialized train dataloader")

    # we load the vocab mapping then
    draft_model.load_vocab_mapping(vocab_mapping_path)
    print_with_rank(f"Loaded vocab mapping")

    if args.eval_data_path is not None:
        eval_eagle3_dataset = build_offline_eagle3_dataset(
            args.eval_hidden_states_path,
            args.max_length,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.batch_size,
            num_workers=4,
            shuffle=False,
            process_group=get_dp_group(),
        )
        print_with_rank(f"Initialized eval dataloader")

    # build Eagle3 model
    # broadcast draft model
    eagle3_model = OfflineEagle3Model(
        target_head=target_head,
        draft_model=draft_model,
        length=args.ttt_length,
    )
    # eagle3_model = DDP(eagle3_model, find_unused_parameters=True)
    eagle3_model = FSDP(
        eagle3_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        ignored_modules=[],
        process_group=get_dp_group(),
    )
    print_with_rank(f"Initialized Eagle3 FSDP model")

    # build other components
    optimizer = torch.optim.AdamW(eagle3_model.parameters(), lr=args.learning_rate)
    total_steps = args.num_epochs * len(train_dataloader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = CosineAnnealingWarmupLR(
        optimizer, total_steps=total_steps, warmup_steps=warmup_steps
    )
    print_with_rank(f"Initialized optimizer and scheduler")

    # start running
    for epoch in range(args.num_epochs):
        # Run training
        train_dataloader.sampler.set_epoch(epoch + 1)
        draft_model.train()
        epoch_acces = [[] for _ in range(eagle3_model.module.length)]
        epoch_plosses = [[] for _ in range(eagle3_model.module.length)]

        for data in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()
            plosses, _, acces = eagle3_model(
                input_ids=data["input_ids"].cuda(),  # [B, S]
                attention_mask=data["attention_mask"].cuda(),  # [B, S]
                loss_mask=data["loss_mask"]
                .unsqueeze(-1)
                .cuda(),  # [B, S, 1] This is different from the online version
                hidden_states=data["hidden_state"].cuda(),  # [B, S, D]
                target=data["target"].cuda(),  # [B, S, D*3]
            )

            # calculate weighted loss
            ploss_weight = [0.8**i for i in range(len(plosses))]
            ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
            ploss.backward()
            optimizer.step()
            scheduler.step()

            logdict = {"train/lr": optimizer.param_groups[0]["lr"]}
            for i in range(len(plosses)):
                logdict[f"train/ploss_{i}"] = plosses[i].item()
            for i in range(len(acces)):
                logdict[f"train/acc_{i}"] = acces[i]
            wandb_log_if_initialized(logdict)

            epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
            epoch_plosses = [
                epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))
            ]

        for i in range(len(epoch_acces)):
            acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
            dist.all_reduce(acc_i)
            acc_i = acc_i / dist.get_world_size()
            acc_i = acc_i.item()
            wandb_log_if_initialized({f"train/epochacc_{i}": acc_i})
            print_on_rank0(
                f"Train Epoch [{epoch + 1}/{args.num_epochs}], position {i},  Acc: {acc_i:.2f}"
            )

        for i in range(len(epoch_plosses)):
            loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
            dist.all_reduce(loss_i)
            loss_i = loss_i / dist.get_world_size()
            loss_i = loss_i.item()
            wandb_log_if_initialized({f"train/epochploss_{i}": loss_i})
            print_on_rank0(
                f"Train Epoch [{epoch + 1}/{args.num_epochs}], position {i}, pLoss: {loss_i:.2f}"
            )

        # run evaluation
        if args.eval_data_path is not None and epoch % args.eval_interval == 0:
            # Run evaluation
            draft_model.eval()
            eval_acces = [[] for _ in range(eagle3_model.length)]
            eval_plosses = [[] for _ in range(eagle3_model.length)]

            for data in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch}"):
                plosses, _, acces = eagle3_model(
                    input_ids=data["input_ids"].cuda(),
                    attention_mask=data["attention_mask"].cuda(),
                    loss_mask=data["loss_mask"].unsqueeze(-1).cuda(),
                    hidden_states=data["hidden_state"].cuda(),
                    target=data["target"].cuda(),
                )
                eval_acces = [eval_acces[i] + [acces[i]] for i in range(len(acces))]
                eval_plosses = [
                    eval_plosses[i] + [plosses[i].item()] for i in range(len(plosses))
                ]

            for i in range(len(epoch_acces)):
                acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
                dist.all_reduce(acc_i)
                acc_i = acc_i / dist.get_world_size()
                acc_i = acc_i.item()

                wandb_log_if_initialized({f"eval/epochacc_{i}": acc_i})
                print_on_rank0(
                    f"Eval Epoch [{epoch + 1}/{args.num_epochs}], position {i},  Acc: {acc_i:.2f}"
                )

            for i in range(len(epoch_plosses)):
                loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
                dist.all_reduce(loss_i)
                loss_i = loss_i / dist.get_world_size()
                loss_i = loss_i.item()

                wandb_log_if_initialized({f"eval/epochploss_{i}": loss_i})
                print_on_rank0(
                    f"Eval Epoch [{epoch + 1}/{args.num_epochs}], position {i}, pLoss: {loss_i:.2f}"
                )

        if epoch % args.save_interval == 0:
            # Save the model
            with FSDP.state_dict_type(eagle3_model, StateDictType.FULL_STATE_DICT):
                model_state_dict = eagle3_model.state_dict()
                draft_model_state_dict = {
                    k.replace("draft_model.", ""): v
                    for k, v in model_state_dict.items()
                    if "draft_model." in k
                }

                if dist.get_rank() == 0:
                    draft_model.save_pretrained(
                        os.path.join(args.output_dir, f"epoch_{epoch}"),
                        state_dict=draft_model_state_dict,
                    )
                dist.barrier()

    destroy_distributed()


if __name__ == "__main__":
    main()
