"""Uneven DSpine supervision must match a globally normalized optimizer step."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from tests.test_modeling.test_dspine import training_inputs, training_model


def fixed_anchors(sequence_length, loss_mask, device, max_valid_anchors=None):
    batch = loss_mask.shape[0]
    return torch.ones(batch, 1, dtype=torch.long, device=device), torch.ones(
        batch, 1, dtype=torch.bool, device=device
    )


def uneven_batch():
    inputs = training_inputs()
    inputs["loss_mask"][0] = torch.tensor([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    inputs["loss_mask"][1] = torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    return inputs


def distributed_step(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=2
    )
    try:
        model = training_model(chunk_size=1)
        model._sample_anchor_positions = fixed_anchors
        inputs = {
            name: tensor[rank : rank + 1] for name, tensor in uneven_batch().items()
        }
        wrapped = DistributedDataParallel(model)
        _, _, metrics = wrapped(**inputs, global_step=300, total_steps=600)
        numerator, denominator = metrics["loss_terms"]
        global_denominator = denominator.clone()
        dist.all_reduce(global_denominator)
        (numerator * 2 / global_denominator).backward()
        torch.save(
            {
                name: parameter.grad
                for name, parameter in model.draft_model.named_parameters()
            },
            Path(directory) / f"gradients-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_gloo_available(), reason="CPU distributed test requires Gloo"
)
def test_uneven_rank_losses_match_one_concatenated_batch(tmp_path):
    model = training_model(chunk_size=1)
    model._sample_anchor_positions = fixed_anchors
    loss, _, _ = model(**uneven_batch(), global_step=300, total_steps=600)
    loss.backward()
    expected = {
        name: parameter.grad for name, parameter in model.draft_model.named_parameters()
    }
    mp.spawn(distributed_step, args=(str(tmp_path),), nprocs=2, join=True)
    for rank in range(2):
        actual = torch.load(tmp_path / f"gradients-{rank}.pt", weights_only=True)
        for name, gradient in expected.items():
            assert gradient is not None and actual[name] is not None, name
            torch.testing.assert_close(actual[name], gradient, atol=2e-5, rtol=2e-4)
