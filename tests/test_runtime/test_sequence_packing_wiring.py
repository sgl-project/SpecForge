"""Packing reaches both offline training topologies and offline evaluation."""

import tempfile
import unittest
from unittest import mock

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.eagle3.data import DataCollatorWithPacking
from specforge.config import Config
from specforge.launch import (
    _make_offline_eval_data_factory,
    _offline_io,
    _streaming_collate,
    build_disagg_offline_runtime,
    build_offline_runtime,
)
from specforge.training.assembly import ModelBundle, build_training_run
from specforge.training.disaggregated import _build_offline, _build_online

ALGORITHM = builtin_algorithm_registry().resolve("eagle3")


def _config():
    return Config.model_validate(
        {
            "model": {
                "target_model_path": "target",
                "draft_model_config": "draft.json",
                "vocab_mapping_path": "mapping.pt",
            },
            "data": {
                "hidden_states_path": "/train-features",
                "eval_hidden_states_path": "/eval-features",
            },
            "training": {"sequence_packing": True, "batch_size": 3, "eval_interval": 1},
        }
    )


def _bundle():
    return ModelBundle(
        model=object(),
        draft_model=object(),
        draft_config=object(),
        target_head=object(),
        strategy_kwargs={},
    )


class SequencePackingWiringTest(unittest.TestCase):
    def test_io_selects_packing_and_preserves_the_normalizer(self):
        packed, normalizer = _offline_io(
            ALGORITHM,
            "text",
            123,
            ttt_length=3,
            use_usp_preprocess=False,
            sequence_packing=True,
        )
        padded, baseline = _offline_io(
            ALGORITHM,
            "text",
            123,
            ttt_length=3,
            use_usp_preprocess=False,
        )
        self.assertIsInstance(packed, DataCollatorWithPacking)
        self.assertNotIsInstance(padded, DataCollatorWithPacking)
        self.assertIs(normalizer.func, baseline.func)
        self.assertEqual(normalizer.keywords, baseline.keywords)

    def test_direct_io_rejects_usp_and_unsupported_provider(self):
        for algorithm, usp in (
            (ALGORITHM, True),
            (builtin_algorithm_registry().resolve("domino"), False),
        ):
            with (
                self.subTest(algorithm=algorithm.name, usp=usp),
                self.assertRaisesRegex(
                    ValueError, "supported non-USP offline provider"
                ),
            ):
                _offline_io(
                    algorithm,
                    "text",
                    123,
                    ttt_length=3,
                    use_usp_preprocess=usp,
                    sequence_packing=True,
                )

    def test_streaming_selects_the_registered_packing_factory(self):
        for name in ("eagle3", "dflash"):
            algorithm = builtin_algorithm_registry().resolve(name)
            collator = _streaming_collate(
                algorithm, "text", None, sequence_packing=True
            )
            self.assertTrue(callable(collator))
        with self.assertRaisesRegex(ValueError, "packing"):
            _streaming_collate(
                builtin_algorithm_registry().resolve("domino"),
                "text",
                None,
                sequence_packing=True,
            )
        with self.assertRaisesRegex(ValueError, "collate"):
            _streaming_collate(
                ALGORITHM, "text", lambda samples: samples, sequence_packing=True
            )

    def test_eval_factory_selects_packing_and_keeps_partial_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            factory = _make_offline_eval_data_factory(
                algorithm=ALGORITHM,
                modality="text",
                hidden_states_path=directory,
                run_id="packing-eval",
                batch_size=3,
                max_len=123,
                ttt_length=3,
                use_usp_preprocess=False,
                dataloader_num_workers=0,
                sequence_packing=True,
            )
        first, second = factory(), factory()
        self.assertIsNot(first, second)
        self.assertIsInstance(first.collate_fn, DataCollatorWithPacking)
        self.assertEqual(first.batch_size, 3)
        self.assertFalse(first.drop_last)

    def test_both_offline_builders_pack_train_and_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            for builder in (build_offline_runtime, build_disagg_offline_runtime):
                with (
                    self.subTest(builder=builder.__name__),
                    mock.patch("specforge.launch._assemble_trainer") as assemble,
                ):
                    data_args = (
                        {"hidden_states_path": directory}
                        if builder is build_offline_runtime
                        else {"feature_store": object(), "refs": []}
                    )
                    builder(
                        algorithm=ALGORITHM,
                        draft_model=object(),
                        target_head=object(),
                        optimizer_factory=object(),
                        run_id="packing-test",
                        output_dir=directory,
                        batch_size=3,
                        accumulation_steps=2,
                        eval_hidden_states_path=directory,
                        sequence_packing=True,
                        **data_args,
                    )
                    kwargs = assemble.call_args.kwargs
                    self.assertIsInstance(kwargs["collate_fn"], DataCollatorWithPacking)
                    self.assertEqual(kwargs["batch_size"], 3)
                    self.assertEqual(kwargs["accumulation_steps"], 2)
                    self.assertIsInstance(
                        kwargs["eval_data_factory"]().collate_fn,
                        DataCollatorWithPacking,
                    )

    def test_unified_offline_assembly_passes_packing(self):
        with (
            mock.patch(
                "specforge.training.assembly.build_model_bundle", return_value=_bundle()
            ),
            mock.patch("specforge.launch.build_offline_runtime") as build,
        ):
            build_training_run(_config(), algorithm=ALGORITHM)
        self.assertTrue(build.call_args.kwargs["sequence_packing"])
        self.assertEqual(build.call_args.kwargs["batch_size"], 3)

    def test_disaggregated_consumer_assembly_passes_packing(self):
        cfg = _config()
        with (
            mock.patch(
                "specforge.training.disaggregated._env", return_value="/manifest.json"
            ),
            mock.patch("specforge.training.disaggregated._wait_for"),
            mock.patch("specforge.training.disaggregated._offline_store"),
            mock.patch(
                "specforge.runtime.data_plane.disagg_ingest.read_ref_manifest",
                return_value=[],
            ),
            mock.patch("specforge.launch.build_disagg_offline_runtime") as build,
        ):
            _build_offline(
                cfg,
                algorithm=ALGORITHM,
                build_model_bundle=lambda _: _bundle(),
                optimizer_factory=lambda _: object(),
                logger=None,
            )
        self.assertTrue(build.call_args.kwargs["sequence_packing"])
        self.assertEqual(build.call_args.kwargs["batch_size"], 3)

    def test_online_consumer_assembly_passes_packing_and_logical_batch_size(self):
        payload = _config().model_dump()
        payload["data"] = {"train_data_path": "train.jsonl"}
        payload["training"].update(role="consumer", max_steps=2, eval_interval=0)
        payload["deployment"] = {
            "mode": "disaggregated",
            "disaggregated": {
                "control_dir": "outputs/packing-test/control",
                "backend": "mooncake",
                "server_urls": ["http://127.0.0.1:30000"],
                "mooncake_metadata_server": "http://127.0.0.1:35880/metadata",
                "mooncake_master_server_addr": "127.0.0.1:35551",
            },
        }
        cfg = Config.model_validate(payload)
        with (
            mock.patch(
                "specforge.training.disaggregated._env",
                return_value="/ref-channel.jsonl",
            ),
            mock.patch("specforge.training.disaggregated._mooncake_store"),
            mock.patch("specforge.launch.build_disagg_online_consumer") as build,
        ):
            _build_online(
                cfg,
                algorithm=ALGORITHM,
                build_model_bundle=lambda _: _bundle(),
                prepare_prompts=lambda *_args, **_kwargs: [],
                optimizer_factory=lambda _: object(),
                logger=None,
            )
        self.assertTrue(build.call_args.kwargs["sequence_packing"])
        self.assertEqual(build.call_args.kwargs["batch_size"], 3)


if __name__ == "__main__":
    unittest.main()
