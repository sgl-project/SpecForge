import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from scripts.prepare_hidden_states import (
    HiddenStatesGenerator,
    _generate_shared_vocab_mapping,
    _resolve_capture_layers,
    _resolve_draft_vocab_size,
    build_target_model,
    parse_args,
    resolve_offline_capture_plan,
)
from specforge.algorithms.builtin import builtin_algorithm_registry

REPO_ROOT = Path(__file__).resolve().parents[2]


class PrepareHiddenStatesCaptureLayersTest(unittest.TestCase):
    def test_cli_defaults_to_legacy_eagle3_capture(self):
        argv = [
            "prepare_hidden_states.py",
            "--target-model-path",
            "target",
            "--data-path",
            "data.jsonl",
        ]
        with mock.patch("sys.argv", argv):
            args = parse_args()

        self.assertEqual("eagle3", args.strategy)
        self.assertIsNone(args.draft_model_config)
        self.assertFalse(hasattr(args, "draft_num_hidden_layers"))
        self.assertFalse(hasattr(args, "draft_block_size"))

    def test_cli_accepts_dflash_family_capture_inputs(self):
        argv = [
            "prepare_hidden_states.py",
            "--target-model-path",
            "target",
            "--data-path",
            "data.jsonl",
            "--strategy",
            "dspark",
            "--draft-model-config",
            "configs/qwen3-4b-dspark.json",
            "--capture-layers",
            "1,9,17,25,33",
        ]
        with mock.patch("sys.argv", argv):
            args = parse_args()

        self.assertEqual("dspark", args.strategy)
        self.assertEqual(
            "configs/qwen3-4b-dspark.json",
            args.draft_model_config,
        )
        self.assertEqual("1,9,17,25,33", args.capture_layers)

    def test_default_and_explicit_capture_layers(self):
        config = SimpleNamespace(num_hidden_layers=32, dtype=None)
        self.assertEqual(_resolve_capture_layers(config, None), [1, 15, 28])
        self.assertEqual(_resolve_capture_layers(config, "2, 7, 19"), [2, 7, 19])

    def test_strategy_capture_plans_use_draft_owned_layers_and_schemas(self):
        expected = {
            "eagle3": (
                "qwen3-8b-eagle3.json",
                (1, 17, 32),
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
        target_config = SimpleNamespace(num_hidden_layers=36)

        for strategy, (config_name, layers, feature_names) in expected.items():
            args = SimpleNamespace(
                strategy=strategy,
                target_model_path="target",
                draft_model_config=str(REPO_ROOT / "configs" / config_name),
                trust_remote_code=False,
                cache_dir=None,
                capture_layers=None,
                output_path="offline-features",
                max_length=128,
            )
            with self.subTest(strategy=strategy):
                plan = resolve_offline_capture_plan(args, target_config)
                self.assertEqual(strategy, plan.strategy)
                self.assertEqual(
                    "eagle3" if strategy == "eagle3" else "dflash",
                    plan.capture_method,
                )
                self.assertEqual(layers, plan.capture_layers)
                self.assertEqual(feature_names, set(plan.layout.output_names))

    def test_non_eagle_capture_layer_override_is_rejected(self):
        args = SimpleNamespace(
            strategy="dflash",
            target_model_path="target",
            capture_layers="1,9,17,25,33",
        )

        with self.assertRaisesRegex(ValueError, "supported only.*eagle3"):
            resolve_offline_capture_plan(
                args,
                SimpleNamespace(num_hidden_layers=36),
            )

    def test_invalid_capture_layers_fail_before_loading_target(self):
        config = SimpleNamespace(num_hidden_layers=32, dtype=None)
        invalid = ("1,2", "1,1,2", "-1,2,3", "1,,3", "one,2,3")
        with mock.patch("scripts.prepare_hidden_states.load_offline_capture") as load:
            for value in invalid:
                args = SimpleNamespace(capture_layers=value)
                with self.subTest(value=value), self.assertRaises(ValueError):
                    build_target_model(args, config)
        load.assert_not_called()

    def test_build_uses_dedicated_offline_loader(self):
        config = SimpleNamespace(num_hidden_layers=32, dtype=None)
        args = SimpleNamespace(
            target_model_path="target",
            trust_remote_code=True,
            capture_layers="2,7,19",
            sglang_attention_backend="fa3",
            sglang_mem_fraction_static=0.4,
            sglang_context_length=4096,
            sglang_enable_nccl_nvls=False,
            sglang_enable_symm_mem=False,
            sglang_enable_torch_compile=False,
            sglang_enable_dp_attention=False,
            sglang_enable_dp_lm_head=False,
            sglang_ep_size=1,
            batch_size=4,
            max_length=128,
        )
        target = mock.Mock()
        with mock.patch(
            "scripts.prepare_hidden_states.load_offline_capture",
            return_value=target,
        ) as load:
            self.assertIs(build_target_model(args, config), target)

        load.assert_called_once()
        self.assertEqual(load.call_args.args, ("target",))
        self.assertNotIn("device", load.call_args.kwargs)
        self.assertNotIn("cache_dir", load.call_args.kwargs)
        target.set_capture_layers.assert_called_once_with(
            [2, 7, 19],
            capture_method="eagle3",
        )

    def test_build_accepts_pre_resolved_arbitrary_capture_layers(self):
        config = SimpleNamespace(num_hidden_layers=40, dtype=None)
        args = SimpleNamespace(
            target_model_path="target",
            trust_remote_code=True,
            capture_layers=None,
            sglang_attention_backend="fa3",
            sglang_mem_fraction_static=0.4,
            sglang_context_length=4096,
            sglang_enable_nccl_nvls=False,
            sglang_enable_symm_mem=False,
            sglang_enable_torch_compile=False,
            sglang_enable_dp_attention=False,
            sglang_enable_dp_lm_head=False,
            sglang_ep_size=1,
            batch_size=4,
            max_length=128,
        )
        target = mock.Mock()
        capture_layers = [1, 9, 17, 25, 33]
        with mock.patch(
            "scripts.prepare_hidden_states.load_offline_capture",
            return_value=target,
        ):
            self.assertIs(
                build_target_model(
                    args,
                    config,
                    capture_layers=capture_layers,
                    capture_method="dflash",
                ),
                target,
            )

        target.set_capture_layers.assert_called_once_with(
            capture_layers,
            capture_method="dflash",
        )


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
    def test_resolves_draft_vocab_size_from_local_json_file(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps({"vocab_size": 16, "draft_vocab_size": 8}),
                encoding="utf-8",
            )

            self.assertEqual(_resolve_draft_vocab_size(str(config_path)), 8)

    def test_rejects_directory_and_hugging_face_repo_id(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "local JSON file"):
                _resolve_draft_vocab_size(directory)
        with self.assertRaisesRegex(FileNotFoundError, "local JSON file"):
            _resolve_draft_vocab_size("org/draft-model")

    def test_resolves_vocab_size_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "draft.json"
            config_path.write_text(json.dumps({"vocab_size": 16}), encoding="utf-8")

            self.assertEqual(_resolve_draft_vocab_size(str(config_path)), 16)

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
