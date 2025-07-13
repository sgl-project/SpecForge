import argparse
import os

import torch
import torch.distributed as dist
import wandb
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sgl_spec import AutoDraftModelConfig, AutoEagle3DraftModel, OnlineEagle3Pipeline
from sgl_spec.data.config import DataConfig, ModelType
from sgl_spec.data.data_pipeline import prepare_full_dataloaders
from sgl_spec.utils import init_distributed


def parse_args():
    parser = argparse.ArgumentParser(description="Train Eagle3 with online data")

    # add model-related arguments
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--draft-model-config", type=str, required=True)

    # add training-related arguments
    parser.add_argument("--train-data-path", type=str, required=True)
    parser.add_argument("--eval-data-path", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)

    # data processing type
    parser.add_argument("--data-type", type=str, default="llama3")

    # other args
    parser.add_argument("--cache-key", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1)

    # wandb wandb args
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-key", type=str, default=None)
    args = parser.parse_args()
    return args


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
    args = parse_args()
    init_distributed()

    if args.wandb and dist.get_rank() == 0:
        init_wandb(args)

    # build target and draft model
    target_model = (
        AutoModelForCausalLM.from_pretrained(args.target_model_path).eval().cuda()
    )
    draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)
    draft_model = AutoEagle3DraftModel.from_config(draft_model_config).cuda()
    draft_model.load_embedding(args.target_model_path)
    draft_model.freeze_embedding()

    eagle3_pipeline = OnlineEagle3Pipeline(
        target_model=target_model,
        draft_model=draft_model,
    )

    # build dataset and dataloader
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    data_config = DataConfig(
        batch_size=args.batch_size,
        model_type=ModelType(args.data_type),
        max_length=args.max_length,
        num_processes=1,
    )

    train_dataloader, eval_dataloader, train_sampler, _, d2t_path = prepare_full_dataloaders(
        tokenizer, args.train_data_path, args.eval_data_path, draft_model=draft_model, config=data_config,
    )

    # build other components
    optimizer = torch.optim.AdamW(draft_model.parameters(), lr=args.learning_rate)

    # start running
    for epoch in range(args.num_epochs):
        # Run training
        train_sampler.set_epoch(epoch)
        draft_model.train()
        epoch_acces = [[] for _ in range(eagle3_pipeline.length)]
        epoch_plosses = [[] for _ in range(eagle3_pipeline.length)]

        for data in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()

            plosses, _, acces = eagle3_pipeline.step(
                input_ids=data["input_ids"].cuda(),
                attention_mask=data["attention_mask"].cuda(),
                loss_mask=data["loss_mask"].cuda(),
            )

            # calculate weighted loss
            ploss_weight = [0.8**i for i in range(len(plosses))]
            ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
            ploss.backward()
            optimizer.step()

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
        if epoch % args.eval_interval == 0:
            # Run evaluation
            draft_model.eval()
            eval_acces = [[] for _ in range(eagle3_pipeline.length)]
            eval_plosses = [[] for _ in range(eagle3_pipeline.length)]
            for data in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch}"):
                plosses, _, acces = eagle3_pipeline.step(
                    input_ids=data["input_ids"].cuda(),
                    attention_mask=data["attention_mask"].cuda(),
                    loss_mask=data["loss_mask"].cuda(),
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

                wandb_log_if_initialized({f"test/epochacc_{i}": acc_i})
                print_on_rank0(
                    f"Test Epoch [{epoch + 1}/{args.num_epochs}], position {i},  Acc: {acc_i:.2f}"
                )

            for i in range(len(epoch_plosses)):
                loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
                dist.all_reduce(loss_i)
                loss_i = loss_i / dist.get_world_size()
                loss_i = loss_i.item()

                wandb_log_if_initialized({f"test/epochploss_{i}": loss_i})
                print_on_rank0(
                    f"Test Epoch [{epoch + 1}/{args.num_epochs}], position {i}, pLoss: {loss_i:.2f}"
                )

        if epoch % args.save_interval == 0:
            # Save the model
            if dist.get_rank() == 0:
                draft_model.save_pretrained(
                    os.path.join(args.output_dir, f"epoch_{epoch}")
                )


if __name__ == "__main__":
    main()
