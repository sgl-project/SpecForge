# coding=utf-8
"""Canonical DSpark disaggregated-consumer optimizer-step gate.

The real DSpark producer requires SGLang server capture. This test starts at the
same Mooncake ``SampleRef`` boundary with shape-real synthetic server features,
then exercises the production ref distributor, durable acknowledgements,
DSpark strategy, FSDP forward/backward, and BF16 optimizer step. No network or
external model is required.
"""

import os
import tempfile
import unittest

import torch

from tests.test_runtime.test_mooncake_store import _FakeMooncakeStore

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "DSpark disaggregated optimizer gate requires CUDA")
class TestDSparkDisaggregatedLaunch(unittest.TestCase):
    def test_synthetic_server_features_train_through_canonical_consumer(self):
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29579")

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from specforge.launch import build_disagg_online_consumer
        from specforge.optimizer import BF16Optimizer
        from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
        from specforge.runtime.data_plane.streaming_ref_channel import (
            StreamingRefChannel,
        )
        from specforge.training.strategies.base import DSparkTrainStrategy

        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        hidden, vocab, sequence_length = 64, fx.V, 12
        accumulation_steps = 2
        workdir = tempfile.mkdtemp(prefix="dspark_disagg_")
        model, captured_width = fx.build_dspark(
            workdir,
            hidden=hidden,
            vocab=vocab,
            block_size=4,
            num_anchors=2,
            attention_backend="sdpa",
        )

        backend = _FakeMooncakeStore()
        producer_store = MooncakeFeatureStore(store=backend, store_id="dspark-e2e")
        consumer_store = MooncakeFeatureStore(
            store=backend,
            store_id="dspark-e2e",
            retain_on_release=True,
        )
        channel = StreamingRefChannel(os.path.join(workdir, "refs.jsonl"))
        logged_metrics = []

        trainer = build_disagg_online_consumer(
            strategy="dspark",
            feature_store=consumer_store,
            channel=channel,
            draft_model=model,
            optimizer_factory=lambda module: BF16Optimizer(
                module,
                lr=1e-3,
                max_grad_norm=0.5,
                warmup_ratio=0.0,
                total_steps=1,
            ),
            run_id="dspark-e2e",
            output_dir=os.path.join(workdir, "out"),
            batch_size=1,
            accumulation_steps=accumulation_steps,
            max_steps=1,
            total_steps=1,
            metadata_db_path=os.path.join(workdir, "consumer.sqlite"),
            logger=lambda metrics, _step: logged_metrics.append(metrics),
            log_interval=1,
        )

        strategy = trainer.core.strategy
        self.assertIsInstance(strategy, DSparkTrainStrategy)
        self.assertIsInstance(strategy.trainable_module(), FSDP)
        self.assertEqual(
            {cls.__name__ for cls in trainer.backend.auto_wrap_block_classes},
            {"Qwen3DFlashDecoderLayer"},
        )
        before = [
            parameter.detach().float().clone()
            for parameter in model.draft_model.parameters()
            if parameter.requires_grad
        ]

        generator = torch.Generator().manual_seed(23)
        for index in range(accumulation_steps):
            tensors = {
                "input_ids": torch.randint(
                    1,
                    vocab,
                    (1, sequence_length),
                    generator=generator,
                ),
                "loss_mask": torch.ones(1, sequence_length, dtype=torch.long),
                "hidden_states": torch.randn(
                    1,
                    sequence_length,
                    captured_width,
                    generator=generator,
                    dtype=torch.bfloat16,
                ),
                "target_last_hidden_states": torch.randn(
                    1,
                    sequence_length,
                    hidden,
                    generator=generator,
                    dtype=torch.bfloat16,
                ),
            }
            ref = producer_store.put(
                tensors,
                sample_id=f"dspark-e2e:sample-{index}",
                metadata={
                    "run_id": "dspark-e2e",
                    "strategy": "dspark",
                    "num_tokens": sequence_length,
                    "target_model_version": "synthetic-qwen3",
                    "target_repr": "hidden_state",
                },
            )
            channel.publish(ref)
        channel.close()

        self.assertEqual(trainer.fit(), 1)
        self.assertEqual(trainer.micro_step, accumulation_steps)
        after = [
            parameter.detach().float().clone()
            for parameter in model.draft_model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(
            any(not torch.equal(old, new) for old, new in zip(before, after)),
            "DSpark optimizer step did not update any draft parameter",
        )

        self.assertTrue(logged_metrics)
        for name in ("loss", "ce_loss", "l1_loss", "confidence_loss"):
            self.assertIn(name, logged_metrics[-1])
            self.assertTrue(torch.isfinite(torch.tensor(logged_metrics[-1][name])))
        marker = trainer.dataflow_controller.store.durable_marker()
        self.assertEqual(marker["global_step"], 1)
        self.assertEqual(
            marker["acked"],
            {f"dspark-e2e:sample-{index}" for index in range(accumulation_steps)},
        )
        self.assertTrue(marker["optimizer_durable"])
        self.assertTrue(os.path.exists(channel.path + ".consumer_done"))
        self.assertEqual(backend._d, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
