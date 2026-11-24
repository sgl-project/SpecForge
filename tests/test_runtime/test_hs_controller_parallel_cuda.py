"""Explicitly scheduled real-CUDA checks for concurrent capture landing slots.

Run on one allocated GPU with SGLANG_RUN_SPEC_FEATURE_CUDA_TESTS=1. The event
wrapper gates only the host's completion observation; all CUDA streams, DMA
copies, and underlying events remain real. CPU discovery never initializes CUDA.
"""

import json
import os
import threading
import unittest
from unittest import mock

import torch
from specforge_sglang_capture.hs import (
    CaptureCancelledError,
    CaptureController,
    CapturedSpan,
    ExecutionIdentity,
    FeatureSchema,
    FeatureSpec,
    PrefixFeatureCache,
)


@unittest.skipUnless(
    os.environ.get("SGLANG_RUN_SPEC_FEATURE_CUDA_TESTS") == "1",
    "requires an explicitly allocated CUDA GPU",
)
class ConcurrentCaptureCudaTests(unittest.TestCase):
    def run_pair(self, cancel_first):
        self.assertTrue(torch.cuda.is_available())
        schema = FeatureSchema(
            (
                FeatureSpec("aux", 16, "bfloat16", (1,)),
                FeatureSpec("last_hidden", 16, "bfloat16", ()),
            )
        )
        rows = 4096
        sample_bytes = rows * schema.row_bytes
        cache = PrefixFeatureCache(schema, 256, sample_bytes * 4)
        controller = CaptureController(
            cache,
            schema,
            sample_bytes * 2,
            sample_bytes * 4,
            sample_bytes * 2,
            max_workers=2,
        )
        plans, spans, expected, samples = [], [], [], []
        tokens = tuple(range(rows))
        for i in range(2):
            execution = ExecutionIdentity("cuda-probe", str(i), str(i), 0)
            namespace = f"cuda-{i}"
            lease = cache.lookup_and_pin(namespace, tokens, 0)
            reservation = controller.try_reserve_output(execution, rows, 0, 0)
            self.assertIsNotNone(reservation)
            plans.append(
                controller.freeze_plan(
                    execution, namespace, tokens, rows, 0, lease, reservation
                )
            )
            cpu = ((torch.arange(rows * 16).reshape(rows, 16) % 113) + i).to(
                torch.bfloat16
            )
            expected.append({"aux": cpu.clone(), "last_hidden": cpu + 5})
            values = {name: tensor.to("cuda") for name, tensor in expected[-1].items()}
            producer = torch.cuda.Event()
            producer.record(torch.cuda.current_stream())
            spans.append(CapturedSpan(execution, 0, rows, values, producer))
        # Each worker must enqueue its own real D2H before either completion
        # observation is allowed to return. This exposes overlapping ownership
        # even on devices whose physical DMA engines serialize the transfers.
        submitted = threading.Barrier(3)
        finish = threading.Event()
        real_event = torch.cuda.Event

        class CopyEvent:
            def __init__(self):
                self.event = real_event()

            def record(self, stream):
                self.event.record(stream)

            def synchronize(self):
                submitted.wait(10)
                if not finish.wait(10):
                    raise RuntimeError("CUDA test host gate did not open")
                self.event.synchronize()

        futures = []
        try:
            with mock.patch(
                "specforge_sglang_capture.hs.controller.torch.cuda.Event", CopyEvent
            ):
                futures = [
                    controller.submit(plan, span) for plan, span in zip(plans, spans)
                ]
                spans.clear()
                values = None
                submitted.wait(10)
                self.assertTrue(all(not future.done() for future in futures))
                self.assertEqual(controller.stats()["pinned_bytes"], sample_bytes * 2)
                offsets = [
                    controller._reservations[p.reservation.reservation_id].pinned_offset
                    for p in plans
                ]
                self.assertEqual(abs(offsets[1] - offsets[0]), sample_bytes)
                extra = ExecutionIdentity("cuda-probe", "extra", "extra", 0)
                self.assertIsNone(controller.try_reserve_output(extra, 4, 0, 0))
                drain = None
                if cancel_first:
                    drain = controller.invalidate(
                        plans[0].execution, "real DMA cancellation"
                    )
                    self.assertFalse(drain.done())
                    self.assertEqual(
                        controller.stats()["pinned_bytes"], sample_bytes * 2
                    )
                finish.set()
                for i, future in enumerate(futures):
                    if cancel_first and i == 0:
                        with self.assertRaises(CaptureCancelledError):
                            future.result(10)
                    else:
                        sample = future.result(10)
                        samples.append(sample)
                        for name, tensor in sample.features.items():
                            self.assertTrue(torch.equal(tensor, expected[i][name]))
                if drain is not None:
                    self.assertTrue(drain.wait(2))
                self.assertEqual(controller.stats()["pinned_bytes"], 0)
                self.assertEqual(controller.stats()["device_bytes"], 0)
            for sample in samples:
                sample.release()
            self.assertEqual(
                cache.stats()["ready_blocks"], 16 * (1 if cancel_first else 2)
            )
            stats = controller.stats()
            self.assertEqual(stats["output_bytes"], 0)
            self.assertEqual(stats["quarantined_executions"], 0)
            self.assertEqual(stats["pinned_arena_bytes"], sample_bytes * 2)
            self.assertEqual(stats["pinned_free_bytes"], sample_bytes * 2)
            print(
                json.dumps(
                    {
                        "real_cuda": True,
                        "cancel_first": cancel_first,
                        "concurrently_submitted_d2h": 2,
                        "bitexact": True,
                        "stats": stats,
                    }
                )
            )
        finally:
            finish.set()
            for future in futures:
                try:
                    future.result(15).release()
                except Exception:
                    pass
            for sample in samples:
                sample.release()
            controller.poll_completed()
            controller.poll_drained()
            controller.shutdown()

    def test_two_real_d2h_jobs_are_isolated_and_bitexact(self):
        self.run_pair(False)

    def test_cancelled_real_d2h_slot_stays_owned_until_observed_drain(self):
        self.run_pair(True)


if __name__ == "__main__":
    unittest.main()
