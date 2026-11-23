# coding=utf-8
"""DrafterChannels: the ref channels of two runs driven as one channel."""

import os
import tempfile
import unittest
from dataclasses import replace
from unittest import mock

from specforge.launch import _publish_refs_with_cleanup
from specforge.runtime.data_plane.drafter_channels import DrafterChannels
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel
from tests.test_runtime.test_streaming_ref_channel import _FailingPublishChannel, _ref


def _refs(count):
    """One ref per run for each of *count* prompts, as one capture returns them."""
    return [
        replace(_ref(f"{run_id}:t{index}"), run_id=run_id)
        for index in range(count)
        for run_id in ("a", "b")
    ]


class DrafterChannelsTest(unittest.TestCase):
    def setUp(self):
        root = tempfile.mkdtemp(prefix="drafter_channels_")
        self.paths = {
            run_id: os.path.join(root, run_id, "refs.jsonl") for run_id in ("a", "b")
        }
        self.channels = self._channels()

    def _channels(self, **replaced):
        channels = {
            run_id: StreamingRefChannel(path) for run_id, path in self.paths.items()
        }
        channels.update(replaced)
        return DrafterChannels(channels)

    def _trainer(self, run_id):
        return StreamingRefChannel(self.paths[run_id])

    def test_capture_waits_for_every_trainer_and_takes_the_largest_quantum(self):
        self._trainer("a").publish_consumer_quantum(20)
        self.assertIsNone(self.channels.consumer_quantum())
        self._trainer("b").publish_consumer_quantum(4)
        self.assertEqual(self.channels.consumer_quantum(), 20)

    def test_capture_pauses_on_the_fullest_running_channel(self):
        self.channels.begin_publish(_refs(30)).commit()
        slow, fast = self._trainer("a"), self._trainer("b")
        slow.mark_consumed(20)
        fast.mark_consumed(27)
        self.assertEqual(self.channels.in_flight_remote(), 10)

        slow.mark_consumer_done()
        self.assertEqual(self.channels.in_flight_remote(), 3)
        self.assertFalse(self.channels.consumer_stopped())
        fast.mark_consumer_done()
        self.assertTrue(self.channels.consumer_stopped())

    def test_each_run_reads_its_own_refs_in_capture_order(self):
        transaction = self.channels.begin_publish(_refs(3))
        self.assertEqual(list(self.channels.published_refs()), [])
        transaction.commit()
        self.assertEqual(transaction.unpublished_refs, ())
        for run_id in ("a", "b"):
            self.assertEqual(
                [ref.sample_id for ref in self._trainer(run_id).poll()],
                [f"{run_id}:t0", f"{run_id}:t1", f"{run_id}:t2"],
            )
        self.assertEqual(
            sorted(ref.sample_id for ref in self.channels.published_refs()),
            sorted(ref.sample_id for ref in _refs(3)),
        )

    def test_a_failed_publish_cleans_only_refs_it_did_not_hand_over(self):
        failing = _FailingPublishChannel(self.paths["a"], fail_after=1)
        channels = self._channels(a=failing)
        store = mock.Mock()
        with self.assertRaisesRegex(OSError, "injected publish failure"):
            _publish_refs_with_cleanup(
                channel=channels, feature_store=store, refs=_refs(2)
            )
        self.assertEqual(
            sorted(ref.sample_id for ref in channels.published_refs()), ["a:t0"]
        )
        self.assertEqual(
            store.abort.call_args_list,
            [
                mock.call(sample_id, reason="producer-ref-publication-failed")
                for sample_id in ("a:t1", "b:t0", "b:t1")
            ],
        )

    def test_fsync_failure_preserves_visible_refs_after_an_earlier_batch(self):
        self.channels.begin_publish(_refs(1)).commit()
        store = mock.Mock()
        with (
            mock.patch(
                "specforge.runtime.data_plane.streaming_ref_channel.os.fsync",
                side_effect=[None, None, OSError("injected fsync failure")],
            ),
            self.assertRaisesRegex(OSError, "injected fsync failure"),
        ):
            _publish_refs_with_cleanup(
                channel=self.channels, feature_store=store, refs=_refs(3)[2:]
            )
        self.assertEqual(
            sorted(ref.sample_id for ref in self.channels.published_refs()),
            ["a:t0", "a:t1", "a:t2", "b:t0", "b:t1"],
        )
        store.abort.assert_called_once_with(
            "b:t2", reason="producer-ref-publication-failed"
        )

    def test_released_bytes_add_up_once_every_trainer_reports_them(self):
        self._trainer("a").mark_consumed(2, num_bytes=10)
        self.assertEqual(self.channels.consumed_state_remote(), (2, None))
        self._trainer("b").mark_consumed(1, num_bytes=5)
        self.assertEqual(self.channels.consumed_state_remote(), (3, 15))

    def test_one_failed_trainer_stops_capture_without_a_clean_close(self):
        self._trainer("b").mark_consumer_failed("optimizer failed")
        self.assertTrue(self.channels.consumer_stopped())
        with self.assertRaisesRegex(RuntimeError, "consumer failed: optimizer failed"):
            self.channels.close()
        self.assertFalse(self._trainer("a").is_closed())

        self.channels.fail("producer stopped")
        for run_id in ("a", "b"):
            self.assertEqual(self._trainer(run_id).failure(), "producer stopped")

    def test_close_closes_every_channel(self):
        self.channels.close()
        for run_id in ("a", "b"):
            self.assertTrue(self._trainer(run_id).is_closed())


if __name__ == "__main__":
    unittest.main()
