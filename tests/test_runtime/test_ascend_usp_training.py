"""Compare one distributed EAGLE3.1 training step with dense full-sequence training.

Run on Ascend, e.g. SPECFORGE_DEVICE=npu SPECFORGE_TEST_ULYSSES=2
torchrun --standalone --nproc-per-node=4 -m tests.test_runtime.test_ascend_usp_training.
The remaining ranks form a ring. U=1 and U=world test the pure variants.
"""

import copy
import json
import os
import unittest
from dataclasses import replace

import torch
import torch.distributed as dist

from specforge.algorithms.objectives import build_objective_model
from specforge.distributed import destroy_distributed, init_distributed
from specforge.modeling import build_draft_model
from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.canonical_data import build_normalizer, collate
from specforge.training.sequence_parallel import configure_sequence_parallel
from specforge.training.strategies.canonical import CanonicalTrainStrategy
from tests.test_modeling.test_canonical_eagle3 import install_mapping, model_config


@unittest.skipUnless("SPECFORGE_TEST_ULYSSES" in os.environ, "requires Ascend torchrun")
class AscendTrainingParityTest(unittest.TestCase):
    def test_training_step(self):
        world = int(os.environ["WORLD_SIZE"])
        ulysses = int(os.environ["SPECFORGE_TEST_ULYSSES"])
        self.assertEqual(world % ulysses, 0)
        ring = world // ulysses
        init_distributed(sp_ulysses_size=ulysses, sp_ring_size=ring)
        try:
            self.compare_step(ulysses, ring)
        finally:
            destroy_distributed()

    def compare_step(self, ulysses, ring):
        torch.manual_seed(821)
        device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
        config = model_config(normalize=True, norm_output=True, target_width=32)
        config = replace(
            config,
            attention_configs={
                "full": replace(
                    config.attention_configs["full"],
                    num_attention_heads=8,
                    num_key_value_heads=2,
                    head_dim=64,
                ),
            },
        )
        draft = build_draft_model(config, backend="eager")
        install_mapping(draft)
        runtime = build_objective_model(
            draft,
            options={"ttt_length": 3},
            target_head=torch.nn.Linear(32, config.vocab_size, bias=False),
        )
        reference = copy.deepcopy(runtime).to(device).train()
        reference.attention_backend = "eager"
        runtime = runtime.to(device).train()
        runtime.attention_backend = "usp"
        configure_sequence_parallel(runtime)
        pc = ParallelConfig.from_distributed(
            sp_ulysses_size=ulysses,
            sp_ring_size=ring,
            sharding_strategy="NO_SHARD",
        )
        backend = FSDPTrainingBackend(pc)
        distributed = backend.prepare_model(runtime)
        keys = ("input_ids", "loss_mask", "hidden_states", "target_last_hidden_states")

        def strategy(model):
            return CanonicalTrainStrategy(
                model, name="eagle3", objective="eagle3", required_features=keys
            )

        # Real context has right padding. Uneven shards own distinct amounts of
        # supervision, including labels which are a neighbour's depth-3 lookahead.
        length = 37
        raw = {
            "input_ids": torch.randint(config.vocab_size, (length,)),
            "position_ids": torch.arange(length) * 2 + 13,
            "attention_mask": torch.arange(length) < length - 2,
            "loss_mask": (
                (torch.arange(length) % 9 < 3) & (torch.arange(length) > 5)
            ).long(),
            "hidden_states": torch.randn(
                length, config.spec_config.target_features.input_width
            ),
            "target_last_hidden_states": torch.randn(length, 32),
        }

        def batch(use_usp):
            sample = build_normalizer(
                length, keys=keys, overlap=3, use_usp_preprocess=use_usp
            )(raw)
            return TrainBatch(["parity"], "eagle3", collate([sample], keys=keys))

        with torch.autocast(device_type="npu", dtype=torch.bfloat16):
            expected = strategy(reference).forward_loss(batch(False))
            actual = strategy(distributed).forward_loss(batch(True))
        expected.loss.backward()
        backend.backward(actual.loss)

        # The normal trainer reports SUM(numerator)/SUM(denominator) over ranks.
        global_loss = actual.loss.detach().clone()
        dist.all_reduce(global_loss)
        global_loss /= dist.get_world_size()
        torch.testing.assert_close(global_loss, expected.loss, rtol=3e-3, atol=1e-5)
        for depth in range(3):
            for prefix in ("ploss", "acceptance_rate"):
                key = f"{prefix}_{depth}"
                pair = torch.stack(actual.ratio_metrics[key])
                dist.all_reduce(pair)
                num, den = expected.ratio_metrics[key]
                torch.testing.assert_close(
                    pair[0] / pair[1], num / den, rtol=3e-3, atol=1e-5
                )

        # Keep weights in FP32 under BF16 autocast so small but real parameter
        # updates cannot disappear simply because two BF16 weights round alike.
        parameters = dict(reference.draft_model.named_parameters())
        trainable = [
            (n, p) for n, p in runtime.draft_model.named_parameters() if p.requires_grad
        ]
        before = {name: param.detach().clone() for name, param in trainable}
        worst_error, worst_name = 0.0, ""
        for name, param in trainable:
            lhs, rhs = param.grad.float(), parameters[name].grad.float()
            error = ((lhs - rhs).norm() / rhs.norm().clamp_min(1e-8)).item()
            self.assertLess(error, 0.055, f"{name} relative gradient L2 error")
            if error > worst_error:
                worst_error, worst_name = error, name
        torch.optim.SGD(reference.parameters(), lr=0.1).step()
        torch.optim.SGD(distributed.parameters(), lr=0.1).step()
        for name, param in trainable:
            expected_delta = parameters[name].detach() - before[name]
            actual_delta = param.detach() - before[name]
            rel = (
                actual_delta - expected_delta
            ).norm() / expected_delta.norm().clamp_min(1e-8)
            self.assertLess(rel.item(), 0.055, f"{name} optimizer update")
        if dist.get_rank() == 0:
            print(
                json.dumps(
                    dict(
                        ulysses=ulysses,
                        ring=ring,
                        dense_loss=expected.loss.item(),
                        usp_loss=global_loss.item(),
                        max_gradient_relative_l2=worst_error,
                        worst_parameter=worst_name,
                        trainable_tensors=len(trainable),
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()
