"""CPU contracts for final batching and consumer-indexed feature demand."""
import os
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from specforge.application import resolve_run
from specforge.config import Config
from specforge.config.draft.codec import encode_config
from specforge.launch import _streaming_collate
from specforge.runtime.data_plane.batch_materialization import ConsumerSchedule
from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader
from specforge.runtime.data_plane.feature_store import LocalFeatureStore
from specforge.runtime.data_plane.sample_ref_queue import SampleRefQueue
from specforge.training.assembly import build_model_bundle
from specforge.training.canonical_data import build_collator
from specforge.training.strategies.base import StepContext
from tests.test_config.draft_fixtures import draft_config
from tests.test_runtime.test_canonical_training import target_directory


DRAFT_KEYS = ("input_ids", "hidden_states", "loss_mask")
WITH_TEACHER = (*DRAFT_KEYS, "target_last_hidden_states")


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


def loader(store, refs=None, *, collate_fn=None, direct=True, demand=True, **kwargs):
    with patch.dict(
        os.environ,
        {
            "SPECFORGE_TRAINER_DIRECT_BATCH": str(int(direct)),
            "SPECFORGE_TRAINER_FEATURE_DEMAND": str(int(demand)),
        },
    ):
        return FeatureDataLoader(
            store, refs=refs, batch_size=2, strategy="dflash",
            collate_fn=collate_fn or build_collator(keys=WITH_TEACHER),
            **kwargs,
        )

def streaming_run(root, *, teacher_metrics=False, loss=None, target="target"):
    training = {
        "role": "consumer",
        "max_steps": 1,
        "attention_backend": "eager",
        "objective_options": {"num_anchors": 2, "teacher_metrics": teacher_metrics},
    }
    if loss is not None:
        training["loss"] = {"terms": {"token": {"kind": loss}}}
    cfg = Config.model_validate(
        {
            "model": {
                "target_model_path": str(target),
                "torch_dtype": "float32",
                "use_liger_kernel": False,
            },
            "data": {"prompts_path": "prompts.jsonl"},
            "draft": encode_config(draft_config()),
            "training": training,
            "output_dir": str(Path(root) / "output"),
            "deployment": {
                "mode": "disaggregated",
                "disaggregated": {
                    "control_dir": str(root),
                    "backend": "mooncake",
                    "server_urls": ["http://capture:30000"],
                },
            },
        }
    )
    return resolve_run(cfg)


class TestBatchMaterialization(unittest.TestCase):
    def test_final_batch_matches_collator_and_owns_fresh_storage(self):
        store = RecordingStore()
        refs = samples(store, [3, 5, 3, 5])
        actual = list(loader(store, refs))
        for batch_index, batch in enumerate(actual):
            raw = [store.get(ref)[0] for ref in refs[batch_index * 2:batch_index * 2 + 2]]
            expected = build_collator(keys=WITH_TEACHER)(raw)
            assert batch.metadata["direct_batch"]
            assert set(batch.tensors) == set(expected)
            for name, value in expected.items():
                torch.testing.assert_close(batch.tensors[name], value)
                assert batch.tensors[name].is_contiguous()
        torch.testing.assert_close(
            actual[0].tensors["attention_mask"],
            torch.tensor([[True, True, True, False, False], [True] * 5]),
        )
        assert "attention_mask" not in store.requests[0][1]
        assert actual[0].tensors['hidden_states'].data_ptr() != actual[1].tensors['hidden_states'].data_ptr()

    def test_prefetch_demand_tracks_consumption_across_resume_and_accumulation(self):
        store = RecordingStore()
        refs = samples(store, [2] * 12)
        queue = SampleRefQueue()
        queue.put(refs)
        with TemporaryDirectory() as root:
            run = streaming_run(root, teacher_metrics=True)
        data = loader(
            store, queue=queue, num_workers=3,
            collate_fn=_streaming_collate(run.algorithm, "text", None),
        )
        # Resumed halfway through optimizer step 8 (zero-index), then metric step 10.
        data.configure_consumer_schedule(global_step=8, accumulation_steps=2,
                                         accumulation_remainder=1, log_interval=10,
                                         has_logger=True, max_batches=6)
        batches = list(data)
        assert [batch.metadata['consumer_batch_index'] for batch in batches] == list(range(6))
        assert ['target_last_hidden_states' in batch.tensors for batch in batches] == [False, True, True, False, False, False]
        assert data.perf_counters_snapshot()['direct_batch_batches'] == 6
        assert data.perf_counters_snapshot()['feature_demand_trimmed_batches'] == 4
        assert {
            sample for sample, names in store.requests
            if "target_last_hidden_states" in names
        } == {"2", "3", "4", "5"}
        self.assertEqual(queue.depth(), 0)
        self.assertEqual(queue.in_flight(), 0)

    def test_unknown_schedule_preserves_diagnostic_teacher_features(self):
        store = RecordingStore()
        refs = samples(store, [2, 3])
        with TemporaryDirectory() as root:
            run = streaming_run(root, teacher_metrics=True)
        data = loader(
            store, refs, collate_fn=_streaming_collate(run.algorithm, "text", None)
        )
        assert 'target_last_hidden_states' in next(iter(data)).tensors
        data = loader(
            store, refs, demand=False,
            collate_fn=_streaming_collate(run.algorithm, "text", None),
        )
        data.configure_consumer_schedule(
            global_step=0, accumulation_steps=1, accumulation_remainder=0,
            log_interval=10, has_logger=False,
        )
        self.assertIn("target_last_hidden_states", next(iter(data)).tensors)
        self.assertEqual(data.perf_counters_snapshot()["feature_demand_trimmed_batches"], 0)

    def test_default_ce_drops_unrequested_teacher_feature_on_both_paths(self):
        store = RecordingStore()
        refs = samples(store, [3, 5])
        with TemporaryDirectory() as root:
            run = streaming_run(root)
        collator = _streaming_collate(run.algorithm, "text", None)
        direct = next(iter(loader(store, refs, collate_fn=collator)))
        regular = next(iter(loader(store, refs, collate_fn=collator, direct=False)))
        self.assertTrue(direct.metadata["direct_batch"])
        self.assertEqual(set(direct.tensors), {*DRAFT_KEYS, "attention_mask"})
        torch.testing.assert_close(direct.tensors, regular.tensors)

    def test_only_diagnostic_teacher_may_be_absent_in_strategy_forward(self):
        from tests.test_algorithms.test_objectives import training_batch

        for loss, teacher_metrics, trim in (
            ("ce", True, True),
            ("ce_l1", True, False),
            ("ce_l1", False, False),
        ):
            with self.subTest(loss=loss, teacher_metrics=teacher_metrics), TemporaryDirectory() as temp:
                root = Path(temp)
                run = streaming_run(root, teacher_metrics=teacher_metrics, loss=loss,
                                    target=target_directory(root))
                store = RecordingStore()
                raw = training_batch(draft_config())
                ref = store.put(
                    {key: raw[key] for key in WITH_TEACHER},
                    sample_id="sample", metadata={"run_id": "run", "strategy": "dflash"},
                )
                collator = _streaming_collate(run.algorithm, "text", None)
                data = loader(store, [ref], collate_fn=collator, drop_last=False)
                data.configure_consumer_schedule(
                    global_step=0, accumulation_steps=1, accumulation_remainder=0,
                    log_interval=10, has_logger=True,
                )
                batch = next(iter(data))
                self.assertTrue(batch.metadata["direct_batch"])
                self.assertEqual("target_last_hidden_states" not in batch.tensors, trim)
                self.assertEqual("target_last_hidden_states" not in store.requests[0][1], trim)
                self.assertEqual(data.perf_counters_snapshot()["feature_demand_trimmed_batches"], int(trim))
                bundle = build_model_bundle(run.config, algorithm=run.algorithm)
                strategy = run.algorithm.providers.step.build(
                    bundle.model, **bundle.strategy_kwargs.options
                )
                ctx = StepContext(collect_detailed_metrics=False)
                reference = next(iter(loader(
                    store, [ref], collate_fn=collator, direct=False, drop_last=False
                )))
                device = next(bundle.model.parameters()).device
                devices = [] if device.type == "cpu" else [device.index]
                with torch.random.fork_rng(devices=devices, device_type=device.type):
                    actual_loss = strategy.forward_loss(batch, ctx).loss
                self.assertTrue(torch.isfinite(actual_loss))
                torch.testing.assert_close(actual_loss, strategy.forward_loss(reference, ctx).loss)
                if trim:
                    self.assertTrue(torch.isfinite(strategy.forward_loss(reference, StepContext()).loss))
                missing_teacher = replace(
                    batch,
                    tensors={key: batch.tensors[key] for key in (*DRAFT_KEYS, "attention_mask")},
                )
                with self.assertRaisesRegex(ValueError, "target_last_hidden_states"):
                    strategy.forward_loss(missing_teacher, StepContext() if trim else ctx)

    def test_explicit_attention_mask_and_position_ids_match_regular_path(self):
        store = RecordingStore()
        collator = build_collator(keys=DRAFT_KEYS)
        refs = []
        for length in (2, 4):
            tensors = {
                "input_ids": torch.arange(length).view(1, length),
                "loss_mask": torch.ones((1, length)),
                "hidden_states": torch.ones((1, length, 2)),
                "attention_mask": torch.arange(length).view(1, length) != 1,
                "position_ids": torch.arange(10, 10 + length).view(1, length),
            }
            refs.append(store.put(tensors, sample_id=str(length),
                                  metadata={"run_id": "run", "strategy": "dflash"}))
        direct = next(iter(loader(store, refs, collate_fn=collator)))
        regular = next(iter(loader(store, refs, collate_fn=collator, direct=False)))
        self.assertTrue(direct.metadata["direct_batch"])
        torch.testing.assert_close(direct.tensors, regular.tensors)
        self.assertEqual(direct.tensors["attention_mask"][0].tolist(), [True, False, False, False])
        self.assertEqual(direct.tensors["position_ids"][0].tolist(), [10, 11, 0, 0])

    def test_legacy_metadata_mixed_mask_and_packed_lengths_use_regular_collation(self):
        store = RecordingStore()
        collator = build_collator(keys=DRAFT_KEYS)
        refs = samples(store, [2, 4])
        features, handle = store.get(refs[1])
        store.release(handle)
        masked = store.put(
            {**features, "attention_mask": torch.tensor([[1, 0, 1, 1]])},
            sample_id="masked", metadata={"run_id": "run", "strategy": "dflash"},
        )
        packed = store.put(
            {**features, "lengths": torch.tensor([2, 2])},
            sample_id="packed", metadata={"run_id": "run", "strategy": "dflash"},
        )
        for batch_refs in ([refs[0], masked], [replace(refs[0], feature_specs={})], [packed]):
            with self.subTest(batch_refs=[ref.sample_id for ref in batch_refs]):
                data = loader(store, batch_refs, collate_fn=collator, drop_last=False)
                actual = next(iter(data))
                expected = next(iter(loader(
                    store, batch_refs, collate_fn=collator, direct=False, drop_last=False
                )))
                self.assertFalse(actual.metadata.get("direct_batch"))
                self.assertEqual(data.perf_counters_snapshot()["direct_batch_fallback_batches"], 1)
                torch.testing.assert_close(actual.tensors, expected.tensors)
        with self.assertRaisesRegex(ValueError, "packed documents require batch size one"):
            next(iter(loader(store, [refs[0], packed], collate_fn=collator)))

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

    def test_unknown_h2d_completion_holds_destination_and_pool_pin_until_cuda_drain(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from unittest import mock
        from tests.test_runtime.test_mooncake_store import _FakeMooncakeStore
        from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
        store = MooncakeFeatureStore(store=_FakeMooncakeStore())
        calls = []
        destination = mock.Mock(is_cuda=True, device=torch.device('cuda', 0))
        destination.record_stream.side_effect = lambda stream: calls.append('record')
        destination.copy_.side_effect = lambda *args, **kwargs: calls.append('enqueue')
        stream = mock.Mock()
        stream.synchronize.side_effect = RuntimeError('CUDA completion unknown')
        slot = SimpleNamespace(storage=torch.empty(32, dtype=torch.uint8))
        pool = mock.Mock(kind='pinned')
        pool.acquire.return_value = (slot, True)
        pool.copy_stream.return_value = stream
        store._receive_pool = pool
        pin = mock.Mock()
        with mock.patch.object(torch.cuda, 'current_stream', return_value=mock.Mock()), mock.patch.object(torch.cuda, 'stream', return_value=nullcontext()):
            with self.assertRaisesRegex(RuntimeError, 'completion unknown'):
                store._copy_segment_into(torch.ones((1,2,4)), destination, pin)
        self.assertEqual(calls, ['record', 'enqueue'])
        self.assertIs(store._h2d_quarantined[0]['destination'], destination)
        pin.release.assert_not_called()
        with self.assertRaises(RuntimeError):
            store._drain_h2d_quarantine()
        self.assertEqual(len(store._h2d_quarantined), 1)
        stream.synchronize.side_effect = None
        store._drain_h2d_quarantine()
        pin.release.assert_called_once()
        self.assertFalse(store._h2d_quarantined)
        store._receive_pool = None
        store.close()

    def test_pool_disabled_segmented_client_requires_proof_anchor_before_connect(self):
        from unittest import mock
        from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
        with patch.dict(os.environ, {'SPECFORGE_HS_SEGMENTED':'1', 'SPECFORGE_TRAINER_NODE_POOL':'0'}):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop('SPECFORGE_TRAINER_NODE_POOL_DIR', None)
                with mock.patch('specforge.runtime.data_plane.mooncake_store._connect_store', side_effect=AssertionError('must fail before connecting')):
                    with self.assertRaisesRegex(ValueError, 'terminal-proof anchor'):
                        MooncakeFeatureStore()

    def test_default_disabled_client_does_not_create_or_require_proof_storage(self):
        from tests.test_runtime.test_mooncake_store import _FakeMooncakeStore
        from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
        with patch.dict(os.environ, {'SPECFORGE_HS_SEGMENTED':'0', 'SPECFORGE_TRAINER_NODE_POOL':'0'}):
            os.environ.pop('SPECFORGE_TRAINER_NODE_POOL_DIR', None)
            store = MooncakeFeatureStore(store=_FakeMooncakeStore())
            self.assertIsNone(store._terminal_proofs)
            self.assertIsNone(store._node_pool)
            store.close()
