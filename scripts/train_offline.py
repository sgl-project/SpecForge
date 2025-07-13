import argparse
import logging
import os

import torch
import wandb
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from sgl_spec import OfflineEagleTrainer
from sgl_spec.data.eagle_data_wrapper import (
    EagleDatasetWrapper,
    custom_eagle_collate_fn,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Eagle3 with offline data")
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="sgl-eagle-training",
        help="Wandb project name.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Wandb run name. Defaults to a random name.",
    )

    args = parser.parse_args()
    return args


def main():
    # load config
    args = parse_args()
    config = load_config(args.config)

    # TODO(yinfan98): only use rank 0 to log to wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
    )
    logger.info("Wandb initialized.")

    # load draft model and tokenizer
    draft_model = build_model(config.model.draft)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.target.pretrained_model_name_or_path
    )

    # freeze the lm head
    # TODO: wait for fan to fix modeling
    draft_model.lm_head.requires_grad_(False)

    # build optimizer
    optimizer = build_optimizer(draft_model, config.optimizer)
    criterion = build_criterion(config.criterion)

    # TODO: refactor the dataset and dataloader
    traindataset = EagleDatasetWrapper(config.data.train.path)
    # testdataset = build_dataset_rank(tokenizer, args.testpath)
    # sampler = DistributedSampler(testdataset, num_replicas=world_size, rank=global_rank, shuffle=False)
    # test_loader = DataLoader(testdataset, batch_size=train_config["bs"], sampler=sampler, num_workers=4, pin_memory=True,
    #                         collate_fn=DataCollatorWithPadding())

    # TODO:(yinfan98): maybe problem here, move model to device
    logger.info("Moving draft model to device.")
    draft_model = draft_model.cuda()

    offline_trainer = OfflineEagleTrainer(
        draft_model, config.model.target.pretrained_model_name_or_path, tokenizer
    )

    train_sampler = DistributedSampler(
        traindataset, num_replicas=1, rank=0, shuffle=True
    )
    train_loader = DataLoader(
        traindataset,
        batch_size=1,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_eagle_collate_fn,
    )

    wandb.watch(draft_model, log="all", log_freq=100)

    start_epoch = 0
    global_step = 0

    for epoch in range(start_epoch, config.train.num_epochs):
        train_sampler.set_epoch(epoch)

        logger.info(f"Now training epoch {epoch}")
        draft_model.train()

        total_train_loss = 0
        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{config.train.num_epochs} [Train]"
        )

        # TODO: complete the training loop, evaluation, checkpoint saving, wandb logging, etc.
        for batch_idx, data in enumerate(train_pbar):

            # move data to device
            hidden_states = data["hidden_state"].cuda()
            hidden_states = hidden_states.permute(
                0, 2, 1, 3
            )  # [1, 3, 114, 5120] -> [1, 114, 3, 5120]
            hidden_states = hidden_states.reshape(
                1, -1, 15360
            )  # [1, 114, 3, 5120] -> [1, 114, 15360]

            # make loss mask
            loss_mask = data["loss_mask"][..., None].cuda()

            optimizer.zero_grad()
            loss = offline_trainer.step(
                hidden_states=hidden_states,
                input_ids=data["input_ids"].int().cuda(),
                target_hidden_states=data["target_hidden_states"].cuda(),
                loss_mask=loss_mask,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(draft_model.parameters(), max_norm=0.5)
            optimizer.step()

            total_train_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())

            # wandb
            wandb.log({"train/step_loss": loss.item(), "global_step": global_step})
            global_step += 1

            avg_train_loss = total_train_loss / len(train_loader)

            logger.info(
                f"Epoch {epoch+1} Average Train Loss: {avg_train_loss:.4f} Train Loss: {loss.item():.4f}"
            )

    logger.info(f"Training finished. Saving final model.")
    final_output_dir = os.path.join(config.output_dir, "final_model")
    os.makedirs(final_output_dir, exist_ok=True)

    # 这样保存不会有 DictConfig 问题
    torch.save(
        draft_model.state_dict(), os.path.join(final_output_dir, "pytorch_model.bin")
    )


if __name__ == "__main__":
    main()
