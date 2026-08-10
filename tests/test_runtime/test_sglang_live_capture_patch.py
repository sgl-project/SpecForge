# coding=utf-8
"""Seam tests for the online-live SGLang patch layer (no GPU, no server).

Runs against the installed sglang tree with the ``patches/sglang/online-live``
layer applied (``scripts/apply_sglang_spec_capture_patch.sh --live``); skipped
otherwise. Exercises the sink's live logic in isolation: deterministic
per-request sampling, passthrough synthesis, and the remove-on-failed-POST
cleanup contract."""

import queue
import unittest
from unittest import mock

import torch

try:
    from sglang.srt import spec_capture_sink
except ImportError:  # pragma: no cover - sglang not installed
    spec_capture_sink = None

LIVE_PATCHED = spec_capture_sink is not None and hasattr(
    spec_capture_sink, "maybe_live_spec"
)

CONFIG = {
    "store_id": "store0",
    "run_id": "run0",
    "gen": 1,
    "features": {"aux": "hidden_states", "last_hidden": "target_last_hidden_states"},
    "passthrough": [
        {"name": "input_ids", "source": "tokens"},
        {"name": "loss_mask", "source": "ones"},
    ],
    "max_num_tokens": 8,
}


def _live_sink(sample_rate: float = 1.0):
    sink = spec_capture_sink.SpecCaptureSink(
        aux_layer_ids=[2, 5, 8],
        intake_url="http://intake:8600",
        sample_rate=sample_rate,
    )
    sink._live_config = dict(CONFIG)
    return sink


@unittest.skipUnless(LIVE_PATCHED, "installed sglang lacks the online-live patch")
class LiveSinkSeamTest(unittest.TestCase):
    def test_module_hook_is_none_without_a_sink(self):
        with mock.patch.object(spec_capture_sink, "_SINK", None):
            self.assertIsNone(spec_capture_sink.maybe_live_spec("rid0"))

    def test_live_spec_is_minted_deterministically_from_the_rid(self):
        sink = _live_sink()
        spec = sink.maybe_live_spec("rid-42")
        self.assertEqual(
            spec,
            {
                "store_id": "store0",
                "sample_id": "live-rid-42",
                "gen": 1,
                "replace": False,
                "features": dict(CONFIG["features"]),
                "live": True,
            },
        )
        # Sampling must agree across TP ranks: same rid -> same decision.
        low, high = _live_sink(sample_rate=0.5), _live_sink(sample_rate=0.5)
        for rid in (f"rid-{i}" for i in range(32)):
            self.assertEqual(
                low.maybe_live_spec(rid) is None, high.maybe_live_spec(rid) is None
            )
        self.assertIsNone(_live_sink(sample_rate=0.0).maybe_live_spec("rid-0"))

    def test_no_intake_url_or_config_means_no_capture(self):
        plain = spec_capture_sink.SpecCaptureSink(aux_layer_ids=[2])
        self.assertIsNone(plain.maybe_live_spec("rid-0"))
        unfetched = _live_sink()
        unfetched._live_config = None
        unfetched._live_config_next_fetch = float("inf")  # block a real fetch
        self.assertIsNone(unfetched.maybe_live_spec("rid-0"))

    def test_write_synthesizes_passthrough_and_posts_the_record(self):
        sink = _live_sink()
        tokens = [11, 12, 13]
        aux = torch.randn(3, 4)
        seen = {}

        def fake_put_sample(spec, *, aux, last_hidden):
            seen["spec"] = spec
            return {
                "sample_id": spec["sample_id"],
                "store_id": spec["store_id"],
                "gen": spec["gen"],
                "aux_layer_ids": sink.aux_layer_ids,
                "features": {"hidden_states": {"shape": [1, 3, 4], "dtype": "float32"}},
            }

        with (
            mock.patch.object(sink, "put_sample", side_effect=fake_put_sample),
            mock.patch.object(
                sink, "_post_live_record", return_value=True
            ) as post,
        ):
            sink._write_live_sample(
                sink.maybe_live_spec("rid-0"), aux, None, tokens
            )
        passthrough = {
            item["name"]: item for item in seen["spec"]["passthrough"]
        }
        self.assertEqual(passthrough["input_ids"]["data"], tokens)
        self.assertEqual(passthrough["loss_mask"]["data"], [1, 1, 1])
        self.assertEqual(passthrough["input_ids"]["shape"], [1, 3])
        self.assertEqual(post.call_args.args[0]["num_tokens"], 3)
        self.assertEqual(sink._live_stats["captured"], 1)

    def test_failed_post_removes_every_written_key(self):
        sink = _live_sink()
        removed = []
        result = {
            "sample_id": "live-rid-0",
            "store_id": "store0",
            "gen": 1,
            "aux_layer_ids": [2, 5, 8],
            "features": {
                "hidden_states": {"shape": [1, 2, 4], "dtype": "float32"},
                "input_ids": {"shape": [1, 2], "dtype": "int64"},
            },
        }
        with (
            mock.patch.object(sink, "put_sample", return_value=dict(result)),
            mock.patch.object(sink, "_post_live_record", return_value=False),
            mock.patch.object(sink, "_remove_quiet", side_effect=removed.append),
        ):
            sink._write_live_sample(
                sink.maybe_live_spec("rid-0"), torch.randn(2, 4), None, [1, 2]
            )
        self.assertEqual(
            sorted(removed),
            [
                "store0/live-rid-0/g1/hidden_states",
                "store0/live-rid-0/g1/input_ids",
            ],
        )
        self.assertEqual(sink._live_stats["dropped_post"], 1)

    def test_inconsistent_rows_and_overlong_requests_are_dropped_before_put(self):
        sink = _live_sink()
        with mock.patch.object(sink, "put_sample") as put:
            spec = sink.maybe_live_spec("rid-0")
            sink._write_live_sample(spec, torch.randn(3, 4), None, [1, 2])  # 3 != 2
            sink._write_live_sample(
                spec, torch.randn(9, 4), None, list(range(9))
            )  # > max_num_tokens
            sink._write_live_sample(spec, torch.randn(1, 4), None, [7])  # warmup
        put.assert_not_called()
        self.assertEqual(sink._live_stats["dropped_bad_rows"], 3)

    def test_full_queue_drops_instead_of_blocking(self):
        sink = _live_sink()
        sink._live_queue = queue.Queue(maxsize=1)
        sink._live_queue.put_nowait(("occupied",))
        sink.put_sample_live(
            {"sample_id": "s"}, aux=None, last_hidden=None, tokens=[1]
        )
        self.assertEqual(sink._live_stats["dropped_queue_full"], 1)


if __name__ == "__main__":
    unittest.main()
