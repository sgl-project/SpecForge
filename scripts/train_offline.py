import argparse
from transformers import AutoTokenizer
from tqdm import tqdm
from sgl_eagle.utils import load_config, build_optimizer, build_criterion, build_model
from sgl_eagle import OfflineEagleTrainer


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
    # traindataset = build_dataset_rank(tokenizer, args.trainpath)
    # testdataset = build_dataset_rank(tokenizer, args.testpath)
    # sampler = DistributedSampler(testdataset, num_replicas=world_size, rank=global_rank, shuffle=False)
    # test_loader = DataLoader(testdataset, batch_size=train_config["bs"], sampler=sampler, num_workers=4, pin_memory=True,
    #                         collate_fn=DataCollatorWithPadding())

    # train_sampler = DistributedSampler(traindataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    # train_loader = DataLoader(traindataset, batch_size=train_config["bs"], sampler=train_sampler, num_workers=4,
    #                         pin_memory=True,
    #                         collate_fn=DataCollatorWithPadding())

    for epoch in range(config.train.num_epochs):
        print(f"Now training epoch {epoch}")
        draft_model.train()

        # TODO: complete the training loop, evaluation, checkpoint saving, wandb logging, etc.
        for batch_idx, data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            output = offline_trainer.step(...)

            # calculate weighted loss
            ploss_weight = [0.8 ** i for i in range(len(plosses))]
            ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
            loss = ploss
            loss.backward()
            optimizer.step()

            epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
            epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]

    # save the model
    draft_model.save_pretrained("./output")

if __name__ == "__main__":
    main()