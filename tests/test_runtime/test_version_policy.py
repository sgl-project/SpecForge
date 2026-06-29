# coding=utf-8
"""Two-axis staleness: WeightRegistry / StalenessPolicy / DriftMonitor + gate."""

import unittest

from specforge.runtime.contracts import SampleRef, WeightVersion
from specforge.runtime.control_plane.version_policy import (
    DriftMonitor,
    StalenessGatedQueue,
    StalenessPolicy,
    WeightRegistry,
)


def _ref(sid, draft="d1", target="t1"):
    return SampleRef(
        sample_id=sid,
        run_id="r",
        source_task_id=None,
        feature_store_uri=f"mem://{sid}",
        feature_keys={},
        feature_specs={},
        strategy="eagle3",
        target_model_version=target,
        draft_weight_version=draft,
    )


def _wv(vid, draft, target="t1"):
    return WeightVersion(
        version_id=vid, draft_weight_version=draft, target_model_version=target
    )


class _FakeQueue:
    def __init__(self, refs):
        self._refs = list(refs)
        self.acked = []

    def get(self, n, timeout_s=0.0):
        out, self._refs = self._refs[:n], self._refs[n:]
        return out

    def ack(self, refs):
        self.acked.extend(refs)

    def fail(self, refs, reason, retryable):
        pass

    def depth(self):
        return len(self._refs)


class _FakeStore:
    def __init__(self):
        self.aborted = []

    def abort(self, sample_id, reason=""):
        self.aborted.append(sample_id)


class TestWeightRegistry(unittest.TestCase):
    def test_draft_lag_by_publish_order(self):
        reg = WeightRegistry()
        reg.publish(_wv("v1", "d1"))
        reg.publish(_wv("v2", "d2"))
        reg.publish(_wv("v3", "d3"))
        self.assertEqual(reg.draft_lag("d3"), 0)  # newest
        self.assertEqual(reg.draft_lag("d2"), 1)
        self.assertEqual(reg.draft_lag("d1"), 2)
        self.assertIsNone(reg.draft_lag("d-unknown"))
        self.assertEqual(reg.latest().version_id, "v3")

    def test_publish_idempotent(self):
        reg = WeightRegistry()
        reg.publish(_wv("v1", "d1"))
        reg.publish(_wv("v1", "d1"))  # same id -> no duplicate
        self.assertEqual(len(reg.history()), 1)


class TestStalenessPolicy(unittest.TestCase):
    def setUp(self):
        self.reg = WeightRegistry()
        self.reg.publish(_wv("v1", "d1"))
        self.reg.publish(_wv("v2", "d2"))  # d2 is newest

    def _assess(self, policy, draft, target="t1"):
        return policy.assess(
            sample_draft_version=draft,
            sample_target_version=target,
            registry=self.reg,
            current_target_version="t1",
        )

    def test_draft_axis_rejects_too_lagged(self):
        p = StalenessPolicy(max_draft_lag=0)
        self.assertTrue(self._assess(p, "d2").accept)  # lag 0
        a = self._assess(p, "d1")  # lag 1 > 0
        self.assertFalse(a.accept)
        self.assertIn("draft_lag>0", a.reasons)

    def test_draft_axis_allows_within_bound(self):
        p = StalenessPolicy(max_draft_lag=1)
        self.assertTrue(self._assess(p, "d1").accept)  # lag 1 <= 1

    def test_unknown_draft_is_maximally_stale(self):
        p = StalenessPolicy(max_draft_lag=5)
        a = self._assess(p, "d-unknown")
        self.assertFalse(a.accept)
        self.assertIn("unknown_draft_version", a.reasons)

    def test_target_axis_rejects_mismatch(self):
        p = StalenessPolicy(require_target_match=True)
        a = self._assess(p, "d2", target="t0")
        self.assertFalse(a.accept)
        self.assertTrue(a.target_stale)
        self.assertIn("target_version_mismatch", a.reasons)

    def test_no_bounds_accepts_everything(self):
        p = StalenessPolicy(max_draft_lag=None, require_target_match=False)
        self.assertTrue(self._assess(p, "d-unknown", target="t9").accept)


class TestDriftMonitor(unittest.TestCase):
    def test_snapshot_and_drifting(self):
        d = DriftMonitor(window=4)
        for lag in (0, 2, 4):
            d.observe(lag)
        d.observe(None)  # unknown version
        snap = d.snapshot()
        self.assertEqual(snap["samples"], 4)
        self.assertEqual(snap["unknown_version"], 1)
        self.assertEqual(snap["max_lag"], 4)
        self.assertAlmostEqual(snap["mean_lag"], 2.0)
        self.assertTrue(d.drifting(mean_lag_threshold=1.0))
        self.assertFalse(d.drifting(mean_lag_threshold=3.0))


class TestStalenessGatedQueue(unittest.TestCase):
    def test_drops_stale_aborts_features_passes_fresh(self):
        reg = WeightRegistry()
        reg.publish(_wv("v1", "d1"))
        reg.publish(_wv("v2", "d2"))  # newest draft d2
        policy = StalenessPolicy(max_draft_lag=0, require_target_match=True)
        inner = _FakeQueue(
            [
                _ref("s0", "d2", "t1"),  # fresh
                _ref("s1", "d1", "t1"),  # draft-stale (lag 1)
                _ref("s2", "d2", "t0"),  # target-stale
            ]
        )
        store = _FakeStore()
        drift = DriftMonitor()
        gate = StalenessGatedQueue(
            inner,
            feature_store=store,
            registry=reg,
            policy=policy,
            current_target_version="t1",
            drift=drift,
        )
        got = gate.get(3)
        self.assertEqual([r.sample_id for r in got], ["s0"])  # only fresh passes
        self.assertEqual(sorted(store.aborted), ["s1", "s2"])  # stale freed
        self.assertEqual(sorted(r.sample_id for r in inner.acked), ["s1", "s2"])
        self.assertEqual(gate.dropped, 2)
        self.assertEqual(drift.snapshot()["samples"], 3)  # all assessed


if __name__ == "__main__":
    unittest.main()
