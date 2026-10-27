"""LongSpark additions to the versioned SGLang capture patch."""

import os
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

from tests.test_runtime.test_spec_capture_sink import (
    _added_source,
    _BufferStore,
    _load_sink,
)


class TestLongSparkCapture(unittest.TestCase):
    def test_selected_kv_rows_are_independent_and_last_hidden_is_post_norm(self):
        module = ModuleType("capture_longspark")
        exec(
            _added_source("python/sglang/srt/spec_capture_longspark.py"),
            module.__dict__,
        )
        keys = [torch.randn(20, 2, 4) for _ in range(5)]
        values = [torch.randn_like(k) for k in keys]
        pool = SimpleNamespace(
            get_key_buffer=keys.__getitem__, get_value_buffer=values.__getitem__
        )
        runner = SimpleNamespace(
            server_args=SimpleNamespace(spec_capture_aux_layer_ids=list(range(5))),
            model_config=SimpleNamespace(hf_config=SimpleNamespace(hidden_size=16)),
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: pool),
        )
        locations = torch.tensor([11, 3, 18])
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: True),
            out_cache_loc=locations,
        )
        logits = SimpleNamespace(
            hidden_states=torch.randn(3, 64), last_hidden_states=torch.randn(3, 16)
        )
        module.capture_longspark_features(
            runner, batch, SimpleNamespace(logits_output=logits)
        )
        self.assertEqual(logits.target_keys.shape, (3, 5, 2, 4))
        expected = torch.stack([k[locations] for k in keys], dim=1)
        for k in keys:
            k.zero_()  # scheduler can now reclaim these source rows
        torch.testing.assert_close(logits.target_keys, expected)
        torch.testing.assert_close(
            logits.hidden_states[:, -16:], logits.last_hidden_states
        )

    def test_capture_validation_uses_the_target_config(self):
        module = ModuleType("capture_longspark")
        exec(
            _added_source("python/sglang/srt/spec_capture_longspark.py"),
            module.__dict__,
        )

        def runner(layers, model_type="qwen3", tp_size=1):
            return SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(
                        model_type=model_type, num_hidden_layers=36
                    )
                ),
                ps=SimpleNamespace(tp_size=tp_size, pp_size=1),
                server_args=SimpleNamespace(spec_capture_aux_layer_ids=layers),
            )

        module.validate_longspark_capture(runner([0, 9, 18, 26, 35]))
        for bad, message in (
            (runner([0, 36]), "aux-layer-ids"),
            (runner([0, 35], model_type="llama"), "Qwen3"),
            (runner([0, 35], tp_size=2), "tp-size"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                module.validate_longspark_capture(bad)

    def test_extra_artifacts_keep_token_major_layout(self):
        with mock.patch.dict(
            os.environ, {"SGLANG_SPEC_CAPTURE_GPU_PUT": "0", "MOONCAKE_PROTOCOL": "tcp"}
        ):
            sink = _load_sink().SpecCaptureSink()
            sink._store = store = _BufferStore()
            try:
                key = torch.randn(3, 5, 2, 4)
                spec = {
                    "store_id": "test",
                    "sample_id": "kv",
                    "features": {
                        "target_keys": "target_keys",
                        "target_values": "target_values",
                    },
                }
                results = sink.submit_samples(
                    [(spec, None, None, {"target_keys": key, "target_values": key + 1})]
                ).result(timeout=5)
                self.assertEqual(
                    results[0]["features"]["target_keys"]["shape"], [1, 3, 5, 2, 4]
                )
                self.assertEqual(
                    store.values["test/kv/g1/target_keys"], key.numpy().tobytes()
                )
                with self.assertRaisesRegex(RuntimeError, "was not captured"):
                    sink.put_samples([(spec, None, None, {})])
            finally:
                sink._executor.shutdown()
