"""DFlash2's own FSDP wrap classes must not shard the directly called selector."""

import functools
import os
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _loss(model, config):
    from tests.test_modeling.test_canonical_dflash2 import model_inputs

    torch.manual_seed(1)
    (ids, features, entries), options = model_inputs(config)
    output = model(ids, features, entries)
    selector = getattr(model, "module", model).candidate_selector
    scores = selector.score_candidates(
        **options,
        unary_logits=output.logits.gather(-1, options["candidate_ids"]),
        hidden_states=output.hidden_states,
    )
    return scores.pow(2).mean() + output.logits.pow(2).mean()


def _build():
    from specforge.modeling import build_draft_model
    from tests.test_modeling.test_canonical_dflash2 import dflash2_config

    torch.manual_seed(0)
    config = dflash2_config()
    model = build_draft_model(config)
    with torch.no_grad():
        model.candidate_selector.successor_codebook.normal_(std=0.1)
    return model, config


def _train_step(rank, world, port, expected):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        model, config = _build()
        classes = {
            type(module)
            for module in model.modules()
            if type(module).__name__ in model._no_split_modules
        }
        wrapped = FSDP(
            model,
            auto_wrap_policy=functools.partial(
                transformer_auto_wrap_policy, transformer_layer_cls=classes
            ),
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            use_orig_params=True,
            device_id=torch.device("cpu"),
        )
        loss = _loss(wrapped, config)
        loss.backward()
        torch.testing.assert_close(loss.detach(), expected)
    finally:
        dist.destroy_process_group()


class CanonicalDFlash2FSDPTest(unittest.TestCase):
    def test_two_rank_shard_grad_op_step_matches_unsharded_loss(self):
        model, config = _build()
        expected = _loss(model, config).detach()
        mp.spawn(_train_step, args=(2, _free_port(), expected), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
