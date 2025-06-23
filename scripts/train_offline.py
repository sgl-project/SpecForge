import argparse
from transformers import AutoTokenizer
from tqdm import tqdm
from sgl_eagle.utils import load_config, build_optimizer, build_criterion, build_model
from sgl_eagle import OfflineEagleTrainer
from sgl_eagle.data.eagle_data_wapper import EagleDatasetWrapper,custom_eagle_collate_fn
from torch.utils.data import DataLoader, DistributedSampler

def parse_args():
    parser = argparse.ArgumentParser(description='Train Eagle3 with online data')
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    # load config
    args = parse_args()
    config = load_config(args.config)


    # load draft model and tokenizer
    draft_model = build_model(config.model.draft)
    tokenizer = AutoTokenizer.from_pretrained(config.model.target.pretrained_model_name_or_path)
    offline_trainer = OfflineEagleTrainer(draft_model, tokenizer)

    # freeze the lm head
    # TODO: wait for fan to fix modeling
    # draft_model.lm_head.requires_grad_(False)

    # build optimizer
    optimizer = build_optimizer(draft_model, config.optimizer)
    criterion = build_criterion(config.criterion)

    # TODO: refactor the dataset and dataloader
    traindataset = EagleDatasetWrapper(args.trainpath)
    # testdataset = build_dataset_rank(tokenizer, args.testpath)
    # sampler = DistributedSampler(testdataset, num_replicas=world_size, rank=global_rank, shuffle=False)
    # test_loader = DataLoader(testdataset, batch_size=train_config["bs"], sampler=sampler, num_workers=4, pin_memory=True,
    #                         collate_fn=DataCollatorWithPadding())

    train_sampler = DistributedSampler(traindataset, num_replicas=1, rank=1, shuffle=True)
    train_loader = DataLoader(traindataset, batch_size=1, sampler=train_sampler, num_workers=4,
                             pin_memory=True,
                             collate_fn=custom_eagle_collate_fn())

    for epoch in range(config.train.num_epochs):
        print(f"Now training epoch {epoch}")
        draft_model.train()

        # TODO: complete the training loop, evaluation, checkpoint saving, wandb logging, etc.
        for batch_idx, data in enumerate(tqdm(train_loader)):
            print("input_ids",data["input_ids"].shape,data["input_ids"])
            import os
            os.exit()
            optimizer.zero_grad()
            loss = offline_trainer.step(
                input_ids=data["input_ids"],
                hidden_state=data["hidden_state"],
                target_hidden_states=data["target_hidden_states"],
            )
            loss.backward()
            optimizer.step()


    # save the model
    draft_model.save_pretrained("./output")

if __name__ == "__main__":
    main()