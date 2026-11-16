"""CPU contracts for final batching and consumer-indexed feature demand."""
import os
from unittest.mock import patch

import unittest
import torch

from specforge.algorithms.common.hidden_states_data import build_collator
from specforge.runtime.data_plane.batch_materialization import ConsumerSchedule
from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader
from specforge.runtime.data_plane.feature_store import LocalFeatureStore


class RecordingStore(LocalFeatureStore):
    def __init__(self):
        super().__init__('direct')
        self.requests = []

    def release(self, handle, *, reason='consumed'):
        with self._lock:
            self._active_leases.pop(handle.lease_token, None)

    def get(self, ref, *, device='cpu', names=None):
        self.requests.append((ref.sample_id, tuple(names or ref.feature_keys)))
        return super().get(ref, device=device, names=names)


def samples(store, lengths):
    refs = []
    for i, length in enumerate(lengths):
        tensors = {
            'input_ids': torch.arange(length).view(1, length),
            'loss_mask': torch.tensor([i % 2] * length).view(1, length),
            'hidden_states': torch.arange(length * 6, dtype=torch.float32).view(1, length, 6) + i,
            'target_last_hidden_states': torch.arange(length * 2, dtype=torch.float32).view(1, length, 2) - i,
        }
        refs.append(store.put(tensors, sample_id=str(i), metadata={'run_id': 'run', 'strategy': 'dflash', 'num_tokens': length}))
    return refs


def loader(store, refs, **kwargs):
    with patch.dict(os.environ, {'SPECFORGE_TRAINER_DIRECT_BATCH': '1',
                                 'SPECFORGE_TRAINER_FEATURE_DEMAND': '1'}):
        return FeatureDataLoader(store, refs=refs, batch_size=2, strategy='dflash',
                                 collate_fn=build_collator(), **kwargs)



class TestBatchMaterialization(unittest.TestCase):
    def test_final_batch_matches_collator_and_owns_fresh_storage(self):
        store = RecordingStore()
        refs = samples(store, [3, 5, 3, 5])
        actual = list(loader(store, refs))
        for batch_index, batch in enumerate(actual):
            raw = [store.get(ref)[0] for ref in refs[batch_index * 2:batch_index * 2 + 2]]
            expected = build_collator()(raw)
            for name, value in expected.items():
                torch.testing.assert_close(batch.tensors[name], value)
                assert batch.tensors[name].is_contiguous()
        assert actual[0].tensors['hidden_states'].data_ptr() != actual[1].tensors['hidden_states'].data_ptr()


    def test_prefetch_demand_tracks_consumption_across_resume_and_accumulation(self):
        store = RecordingStore()
        refs = samples(store, [2] * 12)
        data = loader(store, refs, num_workers=3)
        # Resumed halfway through optimizer step 8 (zero-index), then metric step 10.
        data.configure_consumer_schedule(global_step=8, accumulation_steps=2,
                                         accumulation_remainder=1, log_interval=10,
                                         has_logger=True)
        batches = list(data)
        assert [batch.metadata['consumer_batch_index'] for batch in batches] == list(range(6))
        assert ['target_last_hidden_states' in batch.tensors for batch in batches] == [False, True, True, False, False, False]
        assert data.perf_counters_snapshot()['direct_batch_batches'] == 6
        assert data.perf_counters_snapshot()['feature_demand_trimmed_batches'] == 4


    def test_unknown_schedule_preserves_full_schema_and_other_strategies_do_not_trim(self):
        store = RecordingStore()
        refs = samples(store, [2, 3])
        data = loader(store, refs)
        assert 'target_last_hidden_states' in next(iter(data)).tensors
        data.configure_consumer_schedule(global_step=0, accumulation_steps=1,
                                         accumulation_remainder=0, log_interval=10,
                                         has_logger=False)
        data.strategy = 'dspark'
        from dataclasses import replace
        data._refs = [replace(ref, strategy='dspark') for ref in refs]
        assert 'target_last_hidden_states' in next(iter(data)).tensors


    def test_unknown_transform_falls_back_and_schedule_formula(self):
        store = RecordingStore()
        refs = samples(store, [2, 2])
        data = loader(store, refs, per_sample_transform=lambda tensors: tensors)
        assert not next(iter(data)).metadata.get('direct_batch')
        assert data.perf_counters_snapshot()['direct_batch_fallback_batches'] == 1
        schedule = ConsumerSchedule(9, 3, 0, 10, True)
        assert [schedule.detailed_metrics(i) for i in range(4)] == [True, True, True, False]

    def test_transport_get_into_returns_final_views_and_preserves_padding(self):
        from tests.test_runtime.test_mooncake_store import _FakeMooncakeStore
        from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
        store = MooncakeFeatureStore(store=_FakeMooncakeStore(), store_id='direct',
                                     retain_on_release=True)
        refs = samples(store, [3, 5])
        data = loader(store, refs)
        actual = next(iter(data))
        self.assertEqual(actual.tensors['hidden_states'].shape, (2, 5, 6))
        torch.testing.assert_close(actual.tensors['hidden_states'][0, :3],
                                   torch.arange(18, dtype=torch.float32).view(3, 6))
        self.assertEqual(int(actual.tensors['hidden_states'][0, 3:].count_nonzero()), 0)
        self.assertFalse(store._active_leases)
        store.close()
