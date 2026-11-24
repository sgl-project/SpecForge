"""No-op cache work must disappear without weakening content or ownership checks."""

import hashlib
import struct
import unittest
from dataclasses import replace
from unittest import mock

import torch
from specforge_sglang_capture.hs import (
    CaptureContractError,
    CaptureController,
    CapturedSpan,
    ExecutionIdentity,
    FeatureSchema,
    FeatureSpec,
    PrefixFeatureCache,
)


def schema():
    return FeatureSchema((FeatureSpec("aux", 2, "float32", (1,)),))


class UnusedCacheWorkTests(unittest.TestCase):
    def cache(self, capacity=1024):
        return PrefixFeatureCache(schema(), 4, capacity)

    def test_empty_lease_skips_hash_but_validates_all_tokens_and_identity(self):
        cache = self.cache()
        tokens = tuple(range(8))
        lease = cache.lookup_and_pin("ns", tokens, 0)
        with mock.patch.object(
            cache, "block_keys", side_effect=AssertionError("unnecessary hash")
        ):
            cache._validate_plan_lease(lease, 0, "ns", tokens)
            for bad in [(True,), (-1,), (2**63,), [0]]:
                with self.assertRaises(CaptureContractError):
                    cache._validate_plan_lease(lease, 0, "ns", bad)
            for handle, end, namespace in [
                (replace(lease), 0, "ns"),
                (lease, 1, "ns"),
                (lease, 0, "other"),
                (lease, -1, "ns"),
            ]:
                with self.assertRaises(CaptureContractError):
                    cache._validate_plan_lease(handle, end, namespace, tokens)
            cache.release_lease(lease)
            with self.assertRaises(CaptureContractError):
                cache._validate_plan_lease(lease, 0, "ns", tokens)

    def test_disabled_cache_lookup_validates_namespace_and_tokens_without_content_hash(
        self,
    ):
        cache = self.cache(0)
        with mock.patch.object(
            hashlib, "sha256", side_effect=AssertionError("unnecessary hash")
        ):
            lease = cache.lookup_and_pin("ns", tuple(range(8)), 8)
            self.assertEqual(lease.coverage_end, 0)
            self.assertEqual(cache.stats()["active_leases"], 1)
            cache.release_lease(lease)
            for namespace, tokens, limit in [
                ("", (0,), 1),
                (None, (0,), 1),
                ("ns", (True,), 1),
                ("ns", (0,), -1),
            ]:
                with self.assertRaises(CaptureContractError):
                    cache.lookup_and_pin(namespace, tokens, limit)

    def test_protocol_bytes_and_parent_chain_match_independent_v1_oracle(self):
        tokens = (0, 1, 2**31, 2**63 - 1, 9, 8, 7, 6, 5)
        namespace = "model-测试"
        encoded = namespace.encode()
        parent = hashlib.sha256(
            b"spec-prefix-root-v1\0" + struct.pack("<QQ", len(encoded), 4) + encoded
        ).hexdigest()
        oracle = []
        for start in (0, 4):
            chunk = tokens[start : start + 4]
            body = b"spec-prefix-block-v1\0" + struct.pack(
                "<QQQQQ", len(encoded), 4, start, start + 4, 4
            )
            digest = hashlib.sha256(
                body + encoded + bytes.fromhex(parent) + struct.pack("<4q", *chunk)
            ).hexdigest()
            oracle.append((digest, parent, chunk))
            parent = digest
        keys = self.cache().block_keys(namespace, tokens)
        self.assertEqual([(k.digest, k.parent_digest, k._tokens) for k in keys], oracle)

    def capture(self, cache, n=8):
        controller = CaptureController(cache, schema(), 512, 512, 512)
        execution = ExecutionIdentity("server", "request", "attempt", 0)
        tokens = tuple(range(n))
        lease = cache.lookup_and_pin("ns", tokens, n)
        reservation = controller.try_reserve_output(execution, n, 0, 0)
        plan = controller.freeze_plan(execution, "ns", tokens, n, 0, lease, reservation)
        features = {"aux": torch.arange(n * 2, dtype=torch.float32).reshape(n, 2)}
        span = CapturedSpan(execution, 0, n, features, None)
        with self.assertRaises(CaptureContractError):
            controller.submit(replace(plan, tokens=(99,) * n), span)
        sample = controller.submit(plan, span).result(5)
        self.assertTrue(torch.equal(sample.features["aux"], features["aux"]))
        self.assertEqual(controller.stats()["output_bytes"], n * schema().row_bytes)
        sample.release()
        sample.release()
        controller.poll_completed()
        controller.poll_drained()
        controller.shutdown()
        self.assertEqual(controller.stats()["output_bytes"], 0)
        self.assertEqual(controller.stats()["pending_executions"], 0)
        return controller

    def test_disabled_capture_never_hashes_or_adopts_and_still_drains_once(self):
        cache = self.cache(0)
        with (
            mock.patch.object(
                cache, "block_keys", side_effect=AssertionError("unnecessary hash")
            ),
            mock.patch.object(
                cache,
                "_adopt_released_sample",
                side_effect=AssertionError("unnecessary adoption"),
            ),
        ):
            controller = self.capture(cache)
        self.assertEqual(cache.stats()["ready_blocks"], 0)
        self.assertEqual(controller.stats()["fill_failures"], 0)

    def test_cold_cache_on_still_adopts_and_rejects_wrong_content_lease(self):
        cache = self.cache()
        self.capture(cache)
        self.assertEqual(cache.stats()["ready_blocks"], 2)
        lease = cache.lookup_and_pin("ns", tuple(range(8)), 8)
        self.assertEqual(lease.coverage_end, 8)
        with self.assertRaises(CaptureContractError):
            cache._validate_plan_lease(lease, 4, "ns", (99,) + tuple(range(1, 8)))
        cache.release_lease(lease)

    def test_subchunk_capture_does_not_hash_or_adopt(self):
        cache = self.cache()
        with (
            mock.patch.object(
                cache, "block_keys", side_effect=AssertionError("unnecessary hash")
            ) as keys,
            mock.patch.object(cache, "_adopt_released_sample") as adopt,
        ):
            self.capture(cache, n=3)
        keys.assert_not_called()
        adopt.assert_not_called()
        self.assertEqual(cache.stats()["ready_blocks"], 0)


if __name__ == "__main__":
    unittest.main()
