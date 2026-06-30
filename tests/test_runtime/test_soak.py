# coding=utf-8
"""M5 exit-gate soak: rollout outruns trainer without exhausting storage.

These are the headline M5 gates from the roadmap: "rollout outruns trainer
without exhausting storage" and "sustained-lag soak stays under capacity". They
drive the real loop — feature store + backpressure controller + sample queue +
durable ack — with a fast producer and a deliberately slow consumer, and assert
that tensor residency stays bounded (backpressure throttles the producer) while
the trainer still makes monotonic progress. CPU-only, tiny tensors.
"""

import unittest

import torch

from specforge.runtime.control_plane.backpressure import (
    BackpressureConfig,
    BackpressureController,
)
from specforge.runtime.control_plane.controller import DataFlowController
from specforge.runtime.data_plane.feature_store import LocalFeatureStore
from specforge.runtime.data_plane.sample_ref_queue import SampleRefQueue

_KB = 1024
_SAMPLE_SHAPE = (8, 32)  # 8*32*4 bytes float32 == 1 KiB per sample
_SAMPLE_BYTES = _SAMPLE_SHAPE[0] * _SAMPLE_SHAPE[1] * 4


def _build_pipeline(*, hard_cap, high, low):
    store = LocalFeatureStore("st", max_resident_bytes=hard_cap)
    bp = BackpressureController(
        BackpressureConfig(high_watermark_bytes=high, low_watermark_bytes=low),
        capacity=store,
    )
    ctrl = DataFlowController("run", sample_queue=SampleRefQueue(), backpressure=bp)
    return store, bp, ctrl


def _produce(store, ctrl, task):
    sid = f"sample-{task.task_id}"
    ref = store.put(
        {"h": torch.zeros(*_SAMPLE_SHAPE)},
        sample_id=sid,
        metadata={"source_task_id": task.task_id, "target_repr": "hidden_state"},
    )
    ctrl.commit_samples("w0", [ref])


def _train_one(store, lease, step):
    refs = lease.get(1)
    for ref in refs:
        _tensors, h = store.get(ref)  # materialize
        store.release(h)  # consume-once free (drains residency)
    lease.ack(refs, global_step=step)
    return len(refs)


class TestSoak(unittest.TestCase):
    def test_rollout_outruns_trainer_stays_bounded(self):
        # Producer attempts 4 samples/round, consumer drains 1/round: rollout
        # outruns the trainer 4:1. Residency must never exceed the hard cap and
        # the producer must get throttled (pause transitions > 0).
        hard_cap = 24 * _KB
        store, bp, ctrl = _build_pipeline(hard_cap=hard_cap, high=16 * _KB, low=8 * _KB)
        ctrl.ingest_prompts([{"task_id": f"t{i}", "payload": {}} for i in range(4000)])
        lease = ctrl.train_lease("trainer")

        peak = 0
        rounds = 500
        for r in range(rounds):
            for task in ctrl.lease_prompt_tasks("w0", 4):  # [] when paused
                _produce(store, ctrl, task)
            peak = max(peak, store.health()["resident_bytes"])
            _train_one(store, lease, r)

        # never exhausted storage (no MemoryError thrown; cap never exceeded)
        self.assertLessEqual(peak, hard_cap)
        # backpressure actually engaged (producer outran consumer and got paused)
        self.assertGreaterEqual(bp.snapshot()["pause_transitions"], 1)
        # trainer made real, monotonic progress
        acked = ctrl.status()["durable_acked"]
        self.assertGreater(acked, rounds // 2)
        # backlog of *resident tensors* stayed bounded by the watermark band
        self.assertLessEqual(store.health()["resident_bytes"], hard_cap)

    def test_sustained_lag_soak_drains_to_baseline(self):
        # Trainer lags badly (consumes only every 5th round). Residency must stay
        # under capacity the whole time, then return to baseline once the trainer
        # is allowed to catch up — the leak/soak gate.
        hard_cap = 32 * _KB
        store, bp, ctrl = _build_pipeline(
            hard_cap=hard_cap, high=24 * _KB, low=12 * _KB
        )
        ctrl.ingest_prompts([{"task_id": f"t{i}", "payload": {}} for i in range(6000)])
        lease = ctrl.train_lease("trainer")
        baseline = store.health()["resident_bytes"]

        peak = 0
        for r in range(600):
            for task in ctrl.lease_prompt_tasks("w0", 3):
                _produce(store, ctrl, task)
            if r % 5 == 0:  # trainer is 5x behind
                _train_one(store, lease, r)
            peak = max(peak, store.health()["resident_bytes"])
        self.assertLessEqual(peak, hard_cap)  # never over capacity under lag

        # let the trainer catch up: drain everything still queued
        drained = 0
        while _train_one(store, lease, 10_000 + drained):
            drained += 1
        # all tensors freed -> residency back to baseline (no leak)
        self.assertEqual(store.health()["resident_bytes"], baseline)
        self.assertEqual(store.health()["resident_samples"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
