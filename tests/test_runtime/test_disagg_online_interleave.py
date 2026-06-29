# coding=utf-8
"""O1.2: run_disagg_online_interleaved — producer + trainer run concurrently.

Before O1.2 the colocated online path was drain-then-fit: generate the whole
prompt pool, *then* train. These CPU tests exercise the interleaved driver with
fakes — a real StreamingRefChannel pair, a fake producer-drive that mimics the
real ``drive_producer`` contract (publish + honor should_stop + close in
finally), and a fake trainer that drains a StreamingRefQueue — to prove the three
shutdown paths:

  * full drain then terminate (producer finishes first, closes, loader drains),
  * trainer stops first (cooperative producer wind-down, no hang),
  * producer raises (consumer unblocks + the exception propagates).

No torch / GPU here; the FSDP training half is the GPU launcher test.
"""

import os
import tempfile
import time
import unittest

from specforge.runtime.contracts import SampleRef
from specforge.runtime.data_plane.streaming_ref_channel import (
    StreamingRefChannel,
    StreamingRefQueue,
)
from specforge.runtime.launch import run_disagg_online_interleaved


def _ref(i: int) -> SampleRef:
    sid = f"run0:s{i}"
    return SampleRef(
        sample_id=sid,
        run_id="run0",
        source_task_id=f"t{i}",
        feature_store_uri="mem://run0",
        feature_keys={},
        feature_specs={},
        strategy="eagle3",
        metadata={"target_repr": "logits"},
    )


def _fake_producer(channel, n, *, per_round_sleep=0.0):
    """Mimics drive_producer's contract: publish up to n refs (one per round),
    honor should_stop, and close the channel in finally so the consumer can never
    hang on a finished/dead producer."""
    state = {"produced": 0}

    def drive(max_rounds=1_000_000, should_stop=None):
        try:
            for i in range(min(n, max_rounds)):
                if should_stop is not None and should_stop():
                    break
                channel.publish(_ref(i))
                state["produced"] += 1
                if per_round_sleep:
                    time.sleep(per_round_sleep)
            return state["produced"]
        finally:
            channel.close()

    return drive, state


class _FakeTrainer:
    """Drains a StreamingRefQueue (the 'loader') one ref at a time, up to
    max_steps; records what it consumed. Stands in for TrainerController.fit."""

    def __init__(self, max_steps=None):
        self.max_steps = max_steps
        self.consumed = []

    def fit(self, queue):
        while self.max_steps is None or len(self.consumed) < self.max_steps:
            refs = queue.get(1)
            if not refs:
                break  # channel closed and drained
            self.consumed.extend(r.sample_id for r in refs)
        return len(self.consumed)


class TestInterleavedDriver(unittest.TestCase):
    def _channels(self):
        path = os.path.join(tempfile.mkdtemp(prefix="o12_"), "refs.jsonl")
        # producer + consumer views over the same path (the proven disagg split)
        return StreamingRefChannel(path), StreamingRefChannel(path)

    def test_full_drain_then_terminate(self):
        prod_ch, cons_ch = self._channels()
        drive, state = _fake_producer(prod_ch, 5, per_round_sleep=0.002)
        trainer = _FakeTrainer()
        loader = StreamingRefQueue(cons_ch, poll_s=0.005)

        step = run_disagg_online_interleaved(
            trainer=trainer,
            loader=loader,
            drive_producer=drive,
            channel=prod_ch,
        )
        self.assertEqual(step, 5)
        self.assertEqual(trainer.consumed, [f"run0:s{i}" for i in range(5)])
        self.assertEqual(state["produced"], 5)
        self.assertTrue(prod_ch.is_closed())

    def test_trainer_stops_first_winds_producer_down(self):
        prod_ch, cons_ch = self._channels()
        # producer would publish up to 200 (one per ~3ms); the trainer stops at 3,
        # so the cooperative should_stop must wind the producer down well short.
        drive, state = _fake_producer(prod_ch, 200, per_round_sleep=0.003)
        trainer = _FakeTrainer(max_steps=3)
        loader = StreamingRefQueue(cons_ch, poll_s=0.002)

        step = run_disagg_online_interleaved(
            trainer=trainer,
            loader=loader,
            drive_producer=drive,
            channel=prod_ch,
            join_timeout_s=5.0,
        )
        self.assertEqual(step, 3)
        self.assertEqual(len(trainer.consumed), 3)
        self.assertLess(state["produced"], 200)  # producer did NOT run all rounds

    def test_producer_exception_propagates_and_unblocks_consumer(self):
        prod_ch, cons_ch = self._channels()

        def drive(max_rounds=1_000_000, should_stop=None):
            try:
                prod_ch.publish(_ref(0))
                prod_ch.publish(_ref(1))
                raise RuntimeError("rollout boom")
            finally:
                prod_ch.close()  # the real drive_producer also closes in finally

        trainer = _FakeTrainer()
        loader = StreamingRefQueue(cons_ch, poll_s=0.005)

        with self.assertRaises(RuntimeError) as ctx:
            run_disagg_online_interleaved(
                trainer=trainer,
                loader=loader,
                drive_producer=drive,
                channel=prod_ch,
                join_timeout_s=5.0,
            )
        self.assertIn("rollout boom", str(ctx.exception))
        # the consumer still drained what was published before the crash (no hang)
        self.assertEqual(trainer.consumed, ["run0:s0", "run0:s1"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
