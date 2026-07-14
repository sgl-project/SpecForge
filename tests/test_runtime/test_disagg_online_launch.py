# coding=utf-8
"""Launcher path (ONLINE disaggregated): producer pool -> channel + Mooncake ->
consumer pool, end to end, single rank.

Online analog of test_disagg_launch's FSDP path. The producer drives a
RolloutWorker (HF target, no sglang) that put()s consume-once features into a
MooncakeFeatureStore and streams SampleRefs through a StreamingRefChannel; the
consumer trains from that channel + a SECOND store instance over the same backend
(the disagg topology) through FSDP. Asserts the producer streamed one ref per
prompt, the trainer consumed the whole stream, and consume-once freed the store.

Uses the in-memory Mooncake fake (no master needed); the real cross-node Mooncake
transport is covered by the mooncake_store real test + the 2-node e2e. GPU-only.
"""

import os
import tempfile
import unittest

import torch

from tests.test_runtime.test_mooncake_store import _FakeMooncakeStore

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "online disagg launcher path requires CUDA")
class TestDisaggOnlineLaunch(unittest.TestCase):
    def test_producer_streams_consumer_trains_consume_once(self):
        torch.manual_seed(0)
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29569")

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from specforge.core.eagle3 import OnlineEagle3Model
        from specforge.launch import (
            build_disagg_online_consumer,
            build_disagg_online_producer,
        )
        from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
        from specforge.optimizer import BF16Optimizer
        from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
        from specforge.runtime.data_plane.streaming_ref_channel import (
            StreamingRefChannel,
        )

        H, V, SEQ, TTT, ACC, MAX_OPT_STEPS = fx.H, fx.V, 12, 3, 2, 2
        N = ACC * MAX_OPT_STEPS
        workdir = tempfile.mkdtemp(prefix="disagg_online_")

        target, _dir, aux_ids = fx.build_hf_target(workdir, hidden=H, layers=8, vocab=V)
        cfg = fx.write_draft_config(os.path.join(workdir, "draft.json"))
        vocab_path = fx.write_vocab_mapping(os.path.join(workdir, "vm.pt"))
        draft = AutoDraftModel.from_config(
            AutoDraftModelConfig.from_file(cfg),
            attention_backend="flex_attention",
            torch_dtype=torch.bfloat16,
        ).cuda()
        draft.load_vocab_mapping(vocab_path)
        draft.freeze_embedding()
        eagle3_model = OnlineEagle3Model(
            draft, length=TTT, attention_backend="flex_attention"
        ).cuda()

        g = torch.Generator().manual_seed(7)
        prompts = [
            {
                "payload": {
                    "input_ids": torch.randint(0, V, (SEQ,), generator=g).tolist(),
                    "loss_mask": [1] * SEQ,
                }
            }
            for _ in range(N)
        ]

        # the disagg topology: producer + consumer are SEPARATE store instances
        # over one shared (here, fake-in-memory) Mooncake backend.
        backend = _FakeMooncakeStore()
        producer_store = MooncakeFeatureStore(store=backend, store_id="dol")
        consumer_store = MooncakeFeatureStore(
            store=backend, store_id="dol", retain_on_release=True
        )
        channel = StreamingRefChannel(os.path.join(workdir, "refs.jsonl"))

        # producer pool: rollout -> Mooncake (consume-once) -> channel
        workers, drive_producer = build_disagg_online_producer(
            target_model=target,
            prompts=prompts,
            feature_store=producer_store,
            channel=channel,
            run_id="dol",
            target_hidden_size=H,
            target_vocab_size=V,
            target_repr="logits",
            aux_hidden_state_layer_ids=tuple(aux_ids),
        )

        def optimizer_factory(m):
            return BF16Optimizer(
                m, lr=1e-3, max_grad_norm=0.5, warmup_ratio=0.0, total_steps=10
            )

        # consumer pool: channel + Mooncake -> FSDP train
        trainer = build_disagg_online_consumer(
            feature_store=consumer_store,
            channel=channel,
            draft_model=eagle3_model,
            optimizer_factory=optimizer_factory,
            run_id="dol",
            output_dir=os.path.join(workdir, "out"),
            batch_size=1,
            accumulation_steps=ACC,
            max_steps=MAX_OPT_STEPS,
            metadata_db_path=os.path.join(workdir, "consumer.sqlite"),
        )
        module = trainer.core.strategy.trainable_module()
        self.assertIsInstance(module, FSDP)

        # Consumer setup publishes the optimizer-window contract before capture.
        produced = drive_producer()
        self.assertEqual(produced, N)
        self.assertEqual(channel.published, N)

        step = trainer.fit()
        self.assertEqual(step, MAX_OPT_STEPS)
        self.assertEqual(trainer.micro_step, ACC * MAX_OPT_STEPS)

        marker = trainer.dataflow_controller.store.durable_marker()
        self.assertEqual(len(marker["acked"]), N)
        self.assertEqual(marker["global_step"], MAX_OPT_STEPS)
        self.assertTrue(marker["optimizer_durable"])
        self.assertTrue(os.path.exists(channel.path + ".consumer_done"))
        self.assertFalse(os.path.exists(channel.path + ".consumer_failed"))
        self.assertIsNotNone(trainer.ref_distributor._thread)
        self.assertFalse(trainer.ref_distributor._thread.is_alive())
        # Materialization only releases the read lease. The durable optimizer
        # acknowledgement explicitly removes every feature at the boundary.
        self.assertEqual(backend._d, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
