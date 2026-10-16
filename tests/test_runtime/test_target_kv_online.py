import copy
import importlib.util
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import Qwen3Config

from specforge.inference.adapters.server_capture import (
    ServerCaptureFailure,
    SGLangServerCaptureAdapter,
)
from specforge.inference.capture import CaptureConfig
from specforge.modeling.draft.target_kv import capture_geometry
from specforge.runtime.contracts import SampleRef
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from tests.test_runtime.test_server_capture import (
    _capture_schema,
    _FakeMooncakeStore,
    _StubCaptureServer,
    _task,
)

GEOMETRY = {
    "layer_ids": [2, 5, 8],
    "heads": 2,
    "head_dim": 4,
    "rope_theta": 1000000,
    "rotary_dim": 4,
    "rope_layout": "neox",
}


class OnlineKVTest(unittest.TestCase):
    def test_conv_objective_recipes_are_distinct_and_keep_all_checkpoints(self):
        import json

        from specforge.config import load_config

        root = Path(__file__).resolve().parents[2]
        for objective in ("dpace",):
            path = root / "examples/configs/online/disaggregated/external"
            recipe = load_config(
                str(path / f"qwen3-8b-dspark-kv-conv-{objective}-online.yaml")
            )
            self.assertEqual(recipe.training.max_checkpoints, 0)
            self.assertEqual(recipe.training.dspark_confidence_head_alpha, 0)
            self.assertEqual(recipe.training.strategy, "dspark")
            self.assertEqual(recipe.model.conditioning_source, "kv")
            self.assertEqual(
                recipe.training.loss_type,
                "dflash" if objective == "dpard" else "dpace",
            )
            self.assertFalse(Path(recipe.model.target_model_path).is_absolute())
            draft = json.loads((root / recipe.model.draft_model_config).read_text())
            self.assertTrue(draft["is_causal"])
            self.assertEqual(draft["dflash_config"]["conv_kernel_size"], 2)

    def test_online_recipe_retains_checkpoints(self):
        from specforge.config import load_config

        root = Path(__file__).resolve().parents[2]
        config = load_config(
            str(
                root
                / "examples/configs/online/disaggregated/external/qwen3-8b-dspark-kv-online.yaml"
            )
        )
        self.assertEqual(config.training.max_checkpoints, 0)
        self.assertEqual(config.training.strategy, "dspark")
        self.assertEqual(config.model.conditioning_source, "kv")

    def test_qwen3_geometry_from_dict_and_rope_scaling_rejection(self):
        target = Qwen3Config(
            hidden_size=16,
            vocab_size=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
        )
        draft = dict(
            hidden_size=16,
            vocab_size=32,
            dflash_config={
                "conditioning_source": "target_kv",
                "target_layer_ids": [0, 2],
                "target_kv_heads": 2,
                "target_kv_head_dim": 4,
            },
        )
        self.assertEqual(capture_geometry(draft, target)["rotary_dim"], 4)
        target.rope_parameters = {"rope_type": "linear", "factor": 2}
        with self.assertRaises(ValueError):
            capture_geometry(draft, target)

    def test_kv_transport_and_fail_closed_metadata(self):
        for error in (None, "state", "geometry", "shape"):
            with self.subTest(error=error):
                backend = _FakeMooncakeStore()
                writer = _StubCaptureServer(backend)
                store = MooncakeFeatureStore(store=backend, store_id="run0")

                def server(url, json_body, timeout):
                    rows = writer(url, json_body, timeout)
                    for row in rows:
                        result = row["meta_info"]["spec_capture"]
                        length = result["features"]["input_ids"]["shape"][1]
                        tensor = torch.randn(1, length, 3, 2, 2, 4)
                        key = f"run0/{result['sample_id']}/g1/target_kv"
                        writer._put(key, tensor)
                        result["features"]["target_kv"] = {
                            "shape": list(tensor.shape),
                            "dtype": "float32",
                        }
                        result["target_kv_geometry"] = copy.deepcopy(GEOMETRY)
                        result["target_kv_state"] = "post_qk_norm_post_rope"
                        if error == "state":
                            result["target_kv_state"] = "pre_rope"
                        elif error == "geometry":
                            result["target_kv_geometry"]["layer_ids"] = [1, 5, 8]
                        elif error == "shape":
                            result["features"]["target_kv"]["shape"][-1] = 8
                    return rows

                adapter = SGLangServerCaptureAdapter(
                    "http://test",
                    store,
                    run_id="run0",
                    algorithm="dspark",
                    schema=replace(
                        _capture_schema("dspark", conditioning_source="kv"),
                        target_kv_geometry=GEOMETRY,
                    ),
                    post_fn=server,
                )
                contract = CaptureConfig.from_strategy(
                    required_features={
                        "input_ids",
                        "loss_mask",
                        "target_kv",
                        "target_last_hidden_states",
                    },
                    aux_hidden_state_layer_ids=(2, 5, 8),
                    target_repr="hidden_state",
                    target_hidden_size=8,
                )
                result = adapter.produce_refs([_task("kv", 7)], capture=contract)[0]
                if error:
                    self.assertIsInstance(result, ServerCaptureFailure)
                    self.assertFalse(backend._d)
                else:
                    self.assertIsInstance(result, SampleRef)
                    tensors, handle = store.get(result)
                    self.assertEqual(tensors["target_kv"].shape, (1, 7, 3, 2, 2, 4))
                    store.release(handle)

    def test_server_hooks_capture_owned_post_rope_tensors(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "patches/sglang/v0.5.18/target_kv_capture.py"
        )
        spec = importlib.util.spec_from_file_location("kv_capture_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module._geometry = GEOMETRY
        capture = object.__new__(module.TargetKVCapture)
        capture.layer_ids = (2, 5, 8)
        capture.captured = {}
        batch = SimpleNamespace(
            capture_hidden_mode=SimpleNamespace(is_full=lambda: True),
            forward_mode=SimpleNamespace(is_extend=lambda: True),
        )
        for layer in capture.layer_ids:
            keys, values = torch.full((7, 8), float(layer)), torch.ones(7, 8)
            capture._capture_layer(layer)(None, (None, keys, values, batch))
            keys.zero_()
        hidden = torch.randn(7, 16)
        output = SimpleNamespace()
        capture._output(None, (None, hidden, None, batch), output)
        self.assertEqual(output.hidden_states.shape, (7, 3, 2, 2, 4))
        self.assertEqual(output.hidden_states[0, 0, 0, 0, 0].item(), 2)
        self.assertIs(
            output.last_hidden_states.untyped_storage(), hidden.untyped_storage()
        )
        self.assertFalse(capture.captured)
        with self.assertRaises(ValueError):
            module.capture_metadata({})
        self.assertIn(
            "target_kv_state", module.capture_metadata({"target_kv_geometry": GEOMETRY})
        )


if __name__ == "__main__":
    unittest.main()
