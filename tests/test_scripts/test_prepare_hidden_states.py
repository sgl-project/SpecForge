import unittest
from types import SimpleNamespace
from unittest import mock

from scripts.prepare_hidden_states import _resolve_capture_layers, build_target_model


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


if __name__ == "__main__":
    unittest.main()
