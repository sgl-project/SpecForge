from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from sgl_spec.data.config import DataConfig
from sgl_spec.data.dataset_build import build_dataset_rank, build_test_dataset


class DataCollatorWithPadding:
    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item["input_ids"].shape[1] for item in features)
        batch_input_ids = torch.cat(
            [self.paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [
                self.paddingtensor2D(item["attention_mask"], max_length)
                for item in features
            ]
        )
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item["loss_mask"], max_length) for item in features]
        )
        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


def prepare_dataloaders(
    train_data, test_data, tokenizer, config: Optional[DataConfig] = None
):
    if config is None:
        config = DataConfig()

    # Only compute token dictionary for training data
    train_dataset, token_dict = build_dataset_rank(
        tokenizer, train_data, compute_token_dict=True, config=config
    )

    # Test data doesn't need token counting - use simplified version
    test_dataset, _ = build_test_dataset(tokenizer, test_data, config=config)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=DataCollatorWithPadding(),
    )
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        sampler=test_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=DataCollatorWithPadding(),
    )
    return train_loader, test_loader, train_sampler, test_sampler, token_dict
