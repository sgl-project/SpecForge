import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from scripts.prepare_hidden_states import (
    _generate_shared_vocab_mapping,
    _resolve_capture_layers,
    _resolve_draft_vocab_size,
    build_target_model,
)


class PrepareHiddenStatesCaptureLayersTest(unittest.TestCase):
    def test_default_and_explicit_capture_layers(self):
        config = SimpleNamespace(num_hidden_layers=32, dtype=None)
        self.assertEqual(_resolve_capture_layers(config, None), [1, 15, 28])
        self.assertEqual(_resolve_capture_layers(config, "2, 7, 19"), [2, 7, 19])

    def test_invalid_capture_layers_fail_before_loading_target(self):
        config = SimpleNamespace(num_hidden_layers=32, dtype=None)
        invalid = ("1,2", "1,1,2", "-1,2,3", "1,,3", "one,2,3")
        with mock.patch(
            "scripts.prepare_hidden_states.load_offline_eagle3_capture"
        ) as load:
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
            "scripts.prepare_hidden_states.load_offline_eagle3_capture",
            return_value=target,
        ) as load:
            self.assertIs(build_target_model(args, config), target)

        load.assert_called_once()
        self.assertEqual(load.call_args.args, ("target",))
        self.assertNotIn("device", load.call_args.kwargs)
        self.assertNotIn("cache_dir", load.call_args.kwargs)
        target.set_capture_layers.assert_called_once_with([2, 7, 19])


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
