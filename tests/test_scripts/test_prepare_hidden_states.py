import gzip
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from scripts.prepare_hidden_states import (
    HiddenStatesGenerator,
    _generate_shared_vocab_mapping,
    _resolve_draft_vocab_size,
    parse_args,
    resolve_offline_capture_plan,
)
from specforge.algorithms.builtin import builtin_algorithm_registry

REPO_ROOT = Path(__file__).resolve().parents[2]


class PrepareHiddenStatesCaptureLayersTest(unittest.TestCase):
    def test_cli_defaults_to_legacy_eagle3_capture(self):
        argv = [
            "prepare_hidden_states.py",
            "--server-urls",
            "http://localhost:30000",
            "--target-model-path",
            "target",
            "--data-path",
            "data.jsonl",
        ]
        with mock.patch("sys.argv", argv):
            args = parse_args()

        self.assertEqual("eagle3", args.strategy)
        self.assertIsNone(args.draft_model_config)
        self.assertEqual(args.server_urls, ["http://localhost:30000"])
        self.assertFalse(hasattr(args, "tp_size"))
        self.assertFalse(hasattr(args, "draft_num_hidden_layers"))
        self.assertFalse(hasattr(args, "draft_block_size"))
        self.assertFalse(hasattr(args, "capture_layers"))

    def test_cli_accepts_dflash_family_config(self):
        argv = [
            "prepare_hidden_states.py",
            "--server-urls",
            "http://localhost:30000",
            "--target-model-path",
            "target",
            "--data-path",
            "data.jsonl",
            "--strategy",
            "dspark",
            "--draft-model-config",
            "configs/qwen3-4b-dspark.json",
        ]
        with mock.patch("sys.argv", argv):
            args = parse_args()

        self.assertEqual("dspark", args.strategy)
        self.assertEqual(
            "configs/qwen3-4b-dspark.json",
            args.draft_model_config,
        )
        self.assertFalse(hasattr(args, "capture_layers"))

    def test_strategy_capture_plans_use_draft_owned_layers_and_schemas(self):
        expected = {
            "eagle3": (
                "qwen3-8b-eagle3.json",
                (1, 19, 36),
                {
                    "input_ids",
                    "loss_mask",
                    "aux_hidden_state",
                    "hidden_state",
                },
            ),
            "dflash": (
                "qwen3-8b-dflash.json",
                (1, 9, 17, 25, 33),
                {"input_ids", "loss_mask", "hidden_states"},
            ),
            "domino": (
                "qwen3-8b-domino.json",
                (1, 9, 17, 25, 33),
                {"input_ids", "loss_mask", "hidden_states"},
            ),
            "dspark": (
                "qwen3-4b-dspark.json",
                (1, 9, 17, 25, 33),
                {
                    "input_ids",
                    "loss_mask",
                    "hidden_states",
                    "target_last_hidden_states",
                },
            ),
        }
        target_config = SimpleNamespace(num_hidden_layers=40)

        for strategy, (config_name, layers, feature_names) in expected.items():
            args = SimpleNamespace(
                strategy=strategy,
                target_model_path="target",
                draft_model_config=str(REPO_ROOT / "configs" / config_name),
                trust_remote_code=False,
                cache_dir=None,
                output_path="offline-features",
                max_length=128,
            )
            with self.subTest(strategy=strategy):
                plan = resolve_offline_capture_plan(args, target_config)
                self.assertEqual(strategy, plan.strategy)
                expected_capture_method = {
                    "eagle3": "eagle3",
                    "dspark": "dspark",
                }.get(strategy, "dflash")
                self.assertEqual(expected_capture_method, plan.capture_method)
                self.assertEqual(layers, plan.capture_layers)
                self.assertEqual(feature_names, set(plan.layout.output_names))
                if strategy == "eagle3":
                    self.assertIsNone(plan.loss_mask_filter)
                else:
                    self.assertTrue(plan.loss_mask_filter([0, 1, 1]))
                    self.assertFalse(plan.loss_mask_filter([1, 0, 1]))


class PrepareHiddenStatesSerializationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = builtin_algorithm_registry()

    @staticmethod
    def _generator(*, compress: bool) -> HiddenStatesGenerator:
        generator = HiddenStatesGenerator.__new__(HiddenStatesGenerator)
        generator.compress = compress
        generator.compression_level = 1
        return generator

    @staticmethod
    def _sources():
        return {
            "input_ids": torch.tensor([1, 2, 3]),
            "loss_mask": torch.tensor([1, 1, 0]),
            "aux_hidden_states": torch.arange(120, dtype=torch.float32).reshape(
                1, 3, 5 * 8
            ),
            "last_hidden_states": torch.arange(24, dtype=torch.float32).reshape(
                1, 3, 8
            ),
        }

    def test_saves_each_strategy_schema_compressed_and_uncompressed(self):
        cases = (
            ("eagle3", False),
            ("dflash", True),
            ("domino", False),
            ("dspark", True),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            for strategy, compress in cases:
                with self.subTest(strategy=strategy, compress=compress):
                    provider = self.registry.resolve(strategy).providers.offline_for(
                        "text"
                    )
                    record = provider.capture_layout.materialize(self._sources())
                    suffix = ".ckpt.gz" if compress else ".ckpt"
                    output_file = output_dir / f"{strategy}{suffix}"

                    generator = self._generator(compress=compress)
                    generator._save_tensor_sync(record, str(output_file))

                    if compress:
                        with gzip.open(output_file, "rb") as stream:
                            restored = torch.load(stream, weights_only=True)
                    else:
                        restored = torch.load(output_file, weights_only=True)
                    self.assertEqual(set(record), set(restored))
                    for feature_name, tensor in record.items():
                        self.assertTrue(
                            torch.equal(tensor, restored[feature_name]),
                            feature_name,
                        )

    def test_nan_in_any_strategy_mapped_tensor_skips_the_record(self):
        provider = self.registry.resolve("dflash").providers.offline_for("text")
        sources = self._sources()
        sources["aux_hidden_states"][0, 0, 0] = torch.nan
        record = provider.capture_layout.materialize(sources)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_file = Path(temporary_directory) / "nan.ckpt"
            generator = self._generator(compress=False)
            with mock.patch("builtins.print") as print_warning:
                generator._save_tensor_sync(record, str(output_file))

            self.assertFalse(output_file.exists())
            print_warning.assert_called_once()
            warning = str(print_warning.call_args.args[0])
            self.assertIn("hidden_states", warning)
            self.assertIn(str(output_file), warning)


class PrepareHiddenStatesVocabMappingTest(unittest.TestCase):
    def test_prefers_resolved_draft_vocab_size(self):
        config = SimpleNamespace(vocab_size=16, draft_vocab_size=8)
        self.assertEqual(_resolve_draft_vocab_size(config), 8)

    def test_resolves_vocab_size_fallback(self):
        self.assertEqual(_resolve_draft_vocab_size(SimpleNamespace(vocab_size=16)), 16)

    def test_does_not_fallback_when_draft_vocab_size_is_explicitly_none(self):
        config = SimpleNamespace(vocab_size=16, draft_vocab_size=None)
        with self.assertRaisesRegex(ValueError, "positive"):
            _resolve_draft_vocab_size(config)

    def test_rejects_invalid_resolved_vocab_size(self):
        for value in (None, True, 0, -1, "16"):
            with self.subTest(value=value):
                config = SimpleNamespace(draft_vocab_size=value)
                with self.assertRaisesRegex(ValueError, "positive"):
                    _resolve_draft_vocab_size(config)

    def test_global_rank_zero_generates_fixed_mapping_path(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory) / "vocab_mapping" / "vocab_mapping.pt"

            def generate(**kwargs):
                path = Path(kwargs["cache_dir"]) / f"{kwargs['cache_key']}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "d2t": torch.zeros(4, dtype=torch.long),
                        "t2d": torch.zeros(8, dtype=torch.bool),
                    },
                    path,
                )
                return str(path)

            with (
                mock.patch(
                    "scripts.prepare_hidden_states.generate_vocab_mapping_file",
                    side_effect=generate,
                ) as generate_mock,
                mock.patch(
                    "scripts.prepare_hidden_states.dist.is_initialized",
                    return_value=True,
                ),
                mock.patch(
                    "scripts.prepare_hidden_states.dist.get_rank", return_value=0
                ),
                mock.patch(
                    "scripts.prepare_hidden_states.dist.broadcast_object_list"
                ) as broadcast,
            ):
                actual = _generate_shared_vocab_mapping(
                    object(),
                    output_path=directory,
                    target_vocab_size=8,
                    draft_vocab_size=4,
                )

            self.assertEqual(actual, str(expected))
            self.assertTrue(expected.is_file())
            generate_mock.assert_called_once()
            broadcast.assert_called_once()


if __name__ == "__main__":
    unittest.main()


class CanonicalCaptureTest(unittest.TestCase):
    def test_run_config_captures_what_its_loss_needs(self):
        """--draft resolves the default loss; -c run.yaml carries training.loss."""
        import yaml

        from specforge.config.draft.codec import encode_config
        from tests.test_config.draft_fixtures import draft_config
        from tests.test_runtime.test_canonical_training import target_directory

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = target_directory(root)
            draft = root / "draft.yaml"
            draft.write_text(yaml.safe_dump(encode_config(draft_config())))
            run = root / "run.yaml"
            run.write_text(
                yaml.safe_dump(
                    {
                        "model": {"target_model_path": str(target)},
                        "draft": str(draft),
                        "data": {"hidden_states_path": str(root / "features")},
                        "training": {
                            "attention_backend": "eager",
                            "loss": {"kind": "kl"},
                        },
                    }
                )
            )
            argv = [
                "prepare_hidden_states.py",
                "--server-urls",
                "http://localhost:30000",
                "--data-path",
                "data.jsonl",
            ]
            selections = {
                "draft": ["--draft", str(draft), "--target-model-path", str(target)],
                "config": ["-c", str(run)],
            }
            plans = {}
            for name, selection in selections.items():
                with mock.patch("sys.argv", argv + selection):
                    args = parse_args()
                self.assertEqual(args.target_model_path, str(target))
                plans[name] = resolve_offline_capture_plan(args)
            self.assertIsNone(plans["draft"].layout.last_hidden_feature)
            self.assertEqual(
                plans["config"].layout.last_hidden_feature,
                "target_last_hidden_states",
            )
            other = ["--target-model-path", str(root / "other")]
            with (
                mock.patch("sys.argv", argv + selections["config"] + other),
                mock.patch("sys.stderr"),
                self.assertRaises(SystemExit),
            ):
                parse_args()
            legacy = root / "legacy.yaml"
            legacy.write_text(
                yaml.safe_dump(
                    {
                        "model": {"target_model_path": str(target)},
                        "data": {"hidden_states_path": str(root / "features")},
                    }
                )
            )
            with mock.patch("sys.argv", argv + ["-c", str(legacy)]):
                args = parse_args()
            with self.assertRaisesRegex(ValueError, "not a canonical run"):
                resolve_offline_capture_plan(args)

    def test_eagle_yaml_capture_matches_training_inputs(self):
        import yaml

        from specforge.config.draft.codec import encode_config
        from specforge.training.canonical_data import normalize_sample
        from tests.test_modeling.test_canonical_eagle3 import model_config as eagle
        from tests.test_modeling.test_canonical_peagle import model_config as peagle
        from tests.test_runtime.test_canonical_training import target_directory

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = target_directory(root)
            for factory in (eagle, peagle):
                draft = factory(target_width=32)
                path = root / "draft.yaml"
                path.write_text(yaml.safe_dump(encode_config(draft)))
                argv = [
                    "prepare_hidden_states.py",
                    "--draft",
                    str(path),
                    "--target-model-path",
                    str(target),
                    "--server-urls",
                    "http://localhost:30000",
                    "--data-path",
                    "data.jsonl",
                ]
                with mock.patch("sys.argv", argv):
                    args = parse_args()
                plan = resolve_offline_capture_plan(args)
                self.assertEqual(plan.capture_method, "eagle3")
                self.assertEqual(
                    plan.capture_layers,
                    draft.spec_config.target_features.target_layer_ids,
                )
                record = plan.layout.materialize(
                    {
                        "input_ids": torch.tensor([1, 2, 3]),
                        "loss_mask": torch.tensor([1, 1, 0]),
                        "aux_hidden_states": torch.randn(
                            1, 3, len(plan.capture_layers) * 32
                        ),
                        "last_hidden_states": torch.randn(1, 3, 32),
                    }
                )
                self.assertEqual(
                    set(record),
                    {
                        "input_ids",
                        "loss_mask",
                        "hidden_states",
                        "target_last_hidden_states",
                    },
                )
                sample = normalize_sample(record, keys=tuple(record), max_len=3)
                self.assertEqual(sample["target_last_hidden_states"].shape, (1, 3, 32))
                self.assertEqual(sample["input_ids"].tolist(), [[1, 2, 3]])
