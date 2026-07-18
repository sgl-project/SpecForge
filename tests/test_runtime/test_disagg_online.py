# coding=utf-8
"""Online disaggregation integration (data + control plane, no GPU).

Producer and consumer are SEPARATE MooncakeFeatureStore instances over one shared
backend (the disagg topology). The producer put()s consume-once features and
streams tensor-free refs through a StreamingRefChannel; the consumer's
FeatureDataLoader pulls the stream and materializes batches, freeing each sample
on read (consume-once). Proves the streaming-ref + Mooncake-get + consume-once
free + loader wiring end to end. The FSDP training half is covered by the
cross-node mooncake_store / disagg_launch GPU tests.
"""

import os
import tempfile
import unittest

import torch

from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from specforge.runtime.data_plane.streaming_ref_channel import (
    StreamingRefChannel,
    StreamingRefQueue,
)
from tests.test_runtime.test_mooncake_store import _FakeMooncakeStore, _phys_resident


def _online_features(seed):
    g = torch.Generator().manual_seed(seed)
    return {
        "hidden_state": torch.randn(1, 4, 8, generator=g),
        "target": torch.randn(1, 4, 8, generator=g),
        "loss_mask": torch.ones(1, 4),
    }


def _cat_collate(feats):
    return {k: torch.cat([f[k] for f in feats], dim=0) for k in feats[0]}


class TestOnlineDisaggIntegration(unittest.TestCase):
    def test_stream_to_loader_consume_once(self):
        fake = _FakeMooncakeStore()  # the shared cross-node store
        producer = MooncakeFeatureStore(store=fake, store_id="run0")  # consume-once
        consumer = MooncakeFeatureStore(store=fake, store_id="run0")  # consume-once
        path = os.path.join(tempfile.mkdtemp(prefix="online_disagg_"), "stream.jsonl")

        # producer: stream 4 samples, then close
        wchan = StreamingRefChannel(path)
        srcs = {}
        for i in range(4):
            srcs[f"s{i}"] = _online_features(100 + i)
            ref = producer.put(
                {k: v.clone() for k, v in srcs[f"s{i}"].items()},
                sample_id=f"s{i}",
                metadata={"run_id": "run0", "num_tokens": 4},
            )
            wchan.publish(ref)
        wchan.close()

        # consumer: stream the refs through the loader, materializing consume-once
        rchan = StreamingRefChannel(path)
        queue = StreamingRefQueue(rchan, poll_s=0.0)
        loader = FeatureDataLoader(
            consumer,
            queue,
            batch_size=1,
            collate_fn=_cat_collate,
            drop_last=True,
            strategy="eagle3",
        )

        seen, batches = [], 0
        for batch in loader:
            sid = batch.sample_ids[0]
            seen.append(sid)
            # tensors arrived bit-exact across the (fake) cross-node transport
            self.assertTrue(
                torch.equal(batch.tensors["hidden_state"], srcs[sid]["hidden_state"])
            )
            batches += 1

        self.assertEqual(seen, ["s0", "s1", "s2", "s3"])  # full stream, in order
        self.assertEqual(batches, 4)
        # consume-once: every sample was freed from the shared store on read
        for i in range(4):
            self.assertFalse(_phys_resident(fake, f"s{i}"))
        # backpressure counter advanced as the consumer acked
        self.assertEqual(wchan.consumed_remote(), 4)
        self.assertEqual(wchan.in_flight_remote(), 0)

    def test_loader_blocks_until_producer_closes(self):
        # the loader must NOT end just because the producer is momentarily behind;
        # it ends only once the channel is closed and drained.
        fake = _FakeMooncakeStore()
        producer = MooncakeFeatureStore(store=fake, store_id="run0")
        consumer = MooncakeFeatureStore(store=fake, store_id="run0")
        path = os.path.join(tempfile.mkdtemp(prefix="online_disagg_"), "stream.jsonl")
        wchan = StreamingRefChannel(path)

        # publish 2, but the queue's get() for the 3rd must block until close.
        for i in range(2):
            ref = producer.put(
                _online_features(i), sample_id=f"s{i}", metadata={"run_id": "run0"}
            )
            wchan.publish(ref)

        # a fake clock/sleep: the "sleep" publishes the close so the next poll drains
        state = {"closed": False}

        def sleep(_dt):
            if not state["closed"]:
                wchan.close()
                state["closed"] = True

        rchan = StreamingRefChannel(path)
        queue = StreamingRefQueue(rchan, poll_s=0.1, sleep=sleep)
        loader = FeatureDataLoader(
            consumer,
            queue,
            batch_size=1,
            collate_fn=_cat_collate,
            drop_last=True,
            strategy="eagle3",
        )
        seen = [b.sample_ids[0] for b in loader]
        self.assertEqual(seen, ["s0", "s1"])  # drained then terminated on close


if __name__ == "__main__":
    unittest.main()
