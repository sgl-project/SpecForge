"""Run with torchrun --standalone --nproc_per_node=2; compare USP to full sequence."""

import copy
import os
from dataclasses import replace
from functools import partial

import torch
import torch.distributed as dist
from torch import nn

from specforge.algorithms.objectives import build_objective_model
from specforge.modeling import build_draft_model
from specforge.training.canonical_data import shard_sequence
from specforge.training.sequence_parallel import reduce_counts
from tests.test_modeling.test_canonical_eagle3 import install_mapping, model_config


def main():
    dist.init_process_group("nccl")
    rank, size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(37)
    config = model_config()
    config = replace(
        config,
        attention_configs={
            name: replace(node, is_causal=True)
            for name, node in config.attention_configs.items()
        },
    )
    model = build_draft_model(config)
    install_mapping(model)
    model = model.cuda().to(torch.bfloat16)
    teacher = (
        nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        .cuda()
        .to(torch.bfloat16)
    )
    batch = dict(
        input_ids=torch.arange(16, device="cuda")[None],
        loss_mask=torch.tensor([[0] + [1] * 14 + [0]], device="cuda"),
        attention_mask=torch.ones(1, 16, device="cuda", dtype=torch.long),
        hidden_states=torch.randn(
            1,
            16,
            config.spec_config.target_features.input_width,
            device="cuda",
            dtype=torch.bfloat16,
        ),
        target_last_hidden_states=torch.randn(
            1, 16, config.hidden_size, device="cuda", dtype=torch.bfloat16
        ),
    )
    singles = [dist.new_group([i]) for i in range(size)]
    mixed_u = [dist.new_group([0, 1]), dist.new_group([2, 3])] if size == 4 else None
    mixed_r = [dist.new_group([0, 2]), dist.new_group([1, 3])] if size == 4 else None
    for loss_type in (None, "lambda"):
        for ring in ("mixed",) if size == 4 else (False, True):
            options = dict(ttt_length=3, lk_loss_type=loss_type)
            reference = build_objective_model(
                copy.deepcopy(model),
                target_head=copy.deepcopy(teacher),
                options=options,
            )
            expected = reference(batch).loss
            expected.backward()
            actual = build_objective_model(
                copy.deepcopy(model),
                target_head=copy.deepcopy(teacher),
                options=options,
            )
            actual.attention_backend = "usp"
            actual.ring_group = (
                mixed_r[rank % 2]
                if ring == "mixed"
                else dist.group.WORLD if ring else singles[rank]
            )
            actual.ulysses_group = (
                mixed_u[rank // 2]
                if ring == "mixed"
                else singles[rank] if ring else dist.group.WORLD
            )
            actual.reduce_sequence_metrics = partial(
                reduce_counts, group=dist.group.WORLD
            )
            shard = shard_sequence(batch, rank=rank, size=size, overlap=3)
            shard["position_ids"] = shard["position_ids"].cuda()
            loss = actual(shard).loss
            loss.backward()
            value = loss.detach().clone()
            dist.all_reduce(value)
            torch.testing.assert_close(value / size, expected, atol=0.015, rtol=0.015)
            for (name, p), (_, ref) in zip(
                actual.draft_model.named_parameters(),
                reference.draft_model.named_parameters(),
            ):
                if not p.requires_grad:
                    continue
                grad = p.grad.float()
                dist.all_reduce(grad)
                torch.testing.assert_close(
                    grad / size,
                    ref.grad.float(),
                    atol=0.004,
                    rtol=0.04,
                    msg=lambda msg: f"{name}: {msg}",
                )
            if rank == 0:
                print(
                    f"PASS {'ring' if ring else 'ulysses'} loss={loss_type or 'CE'}",
                    flush=True,
                )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
