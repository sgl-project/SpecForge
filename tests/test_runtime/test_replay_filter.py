"""Fresh producer replay preserves the exact unacknowledged immutable plan."""
import tempfile
import unittest
from pathlib import Path

from specforge.launch import _iter_epoch_online_prompt_batches
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel
from tests.test_runtime.test_disagg_multiserver import _adapter, _build, _prompts
from tests.test_runtime.test_server_capture import _FakeMooncakeStore, _StubCaptureServer


class ReplayFilterTest(unittest.TestCase):
    def test_plan_order_and_complement(self):
        prompts = _prompts(37)
        original = [p for epoch in (1, 2) for batch in _iter_epoch_online_prompt_batches(
            prompts, epoch, 2, seed=42, batch_size=7) for p in batch]
        skipped = {p['task_id'] for p in original[::3]}
        replay = [p for epoch in (1, 2) for batch in _iter_epoch_online_prompt_batches(
            prompts, epoch, 2, seed=42, batch_size=7, skip_task_ids=skipped) for p in batch]
        self.assertEqual(replay, [p for p in original if p['task_id'] not in skipped])
        self.assertEqual(len(replay) + len(skipped), 74)

    def test_fresh_producer_excludes_acked_ids_across_epochs(self):
        backend = _FakeMooncakeStore()
        stub = _StubCaptureServer(backend)
        store = MooncakeFeatureStore(store=backend, store_id='run0')
        expected = {f'run0:epoch{e:04d}-prompt{i:012d}' for e in (1, 2) for i in range(19)}
        acked = {f'run0:epoch0001-prompt{i:012d}' for i in range(0, 19, 2)}
        with tempfile.TemporaryDirectory() as directory:
            channel = StreamingRefChannel(str(Path(directory) / 'refs.jsonl'))
            _, drive = _build([_adapter(store, stub)], _prompts(19), store, channel,
                prompt_epochs=2, prompt_epoch_offset=1, prompt_seed=42,
                prompt_ingest_batch_size=7, lease=2, excluded_sample_ids=acked)
            self.assertEqual(drive(), len(expected - acked))
            actual = [ref.sample_id for ref in StreamingRefChannel(channel.path).poll()]
            self.assertEqual(set(actual), expected - acked)
            self.assertEqual(len(actual), len(set(actual)))

    def test_skipped_rows_are_not_materialized(self):
        class Prompts:
            def __len__(self):
                return 4

            def __getitem__(self, index):
                if index in (1, 3):
                    raise AssertionError('acknowledged payload was read')
                return {'payload': {'input_ids': [index], 'loss_mask': [1]}}

        batches = _iter_epoch_online_prompt_batches(
            Prompts(), 1, 2, batch_size=1,
            skip_task_ids={'epoch0001-prompt000000000001', 'epoch0001-prompt000000000003'})
        tasks = [prompt['task_id'] for batch in batches for prompt in batch]
        self.assertEqual(set(tasks), {'epoch0001-prompt000000000000', 'epoch0001-prompt000000000002'})

    def test_rejects_wrong_run_and_out_of_plan_ids(self):
        backend = _FakeMooncakeStore()
        store = MooncakeFeatureStore(store=backend, store_id='run0')
        stub = _StubCaptureServer(backend)
        for sample_id in ('other:epoch0001-prompt000000000001',
                          'run0:epoch0003-prompt000000000001',
                          'run0:epoch0001-prompt000000000099', 'run0:unversioned'):
            with self.subTest(sample_id=sample_id), tempfile.TemporaryDirectory() as directory:
                channel = StreamingRefChannel(str(Path(directory) / 'refs.jsonl'))
                with self.assertRaises(ValueError):
                    _build([_adapter(store, stub)], _prompts(19), store, channel,
                        prompt_epochs=2, prompt_epoch_offset=1, excluded_sample_ids={sample_id})


if __name__ == '__main__':
    unittest.main()
