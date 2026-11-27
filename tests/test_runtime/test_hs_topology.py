"""CPU startup topology contracts using the real SG ParallelContext getter."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from sglang.srt import runtime_context
from sglang.srt.arg_groups.model_override_base import resolved_view
from specforge_sglang_capture.hs.config import HSConfig
from specforge_sglang_capture.hs.protocol import CaptureContractError
from specforge_sglang_capture.hs.runtime import HSCaptureRuntime


class HSStartupTopologyTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            enable_dp_attention=False,
            enable_hierarchical_cache=False,
            enable_lora=False,
            enable_session_radix_cache=False,
            disable_radix_cache=False,
            enable_memory_saver=False,
            enable_deterministic_inference=True,
            model_path="/cpu-fixture",
            revision=None,
            quantization=None,
            load_format="auto",
            kv_cache_dtype="auto",
            bf16_gemm_backend="auto",
            attention_backend="flashinfer",
            prefill_attention_backend=None,
        )
        text = SimpleNamespace(hidden_size=4, to_dict=lambda: {"hidden_size": 4})
        runtime = HSCaptureRuntime.__new__(HSCaptureRuntime)
        runtime.s = SimpleNamespace(
            server_args=self.args,
            enable_priority_preemption=False,
            prefill_delayer=None,
            dllm_config=None,
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            page_size=1,
            model_config=SimpleNamespace(hf_text_config=text, dtype=torch.float32),
        )
        runtime.config, runtime.enabled = (
            HSConfig(capture=True, cache=True, chunk_tokens=4),
            True,
        )
        self.runtime = runtime
        self.parallel = runtime_context.get_parallel()
        self.assertIsInstance(self.parallel, runtime_context.ParallelContext)
        self.env = mock.patch.dict(
            os.environ, {"MOONCAKE_PROTOCOL": "tcp", "SGLANG_SPEC_CAPTURE_GPU_PUT": "0"}
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def identity(self, **overrides):
        topology = dict(
            tp_size=2,
            tp_rank=0,
            pp_size=1,
            pp_rank=0,
            dp_size=1,
            attn_dp_size=1,
            attn_dp_rank=0,
            attn_cp_size=1,
            attn_cp_rank=0,
            attn_tp_size=2,
            attn_tp_rank=0,
            dcp_size=1,
        )
        topology.update(overrides)
        features = SimpleNamespace(
            return_aux_hidden_state_layer_ids=[1, 3], aux_hidden_state_capture="dflash"
        )
        checkpoint = mock.Mock(return_value={"backend": "isolated-cpu-fixture"})
        checkpoint_module = SimpleNamespace(
            validate_capture_checkpoint_support=checkpoint
        )
        with (
            self.parallel.override(**topology),
            mock.patch.object(
                runtime_context,
                "get_schedule",
                return_value=SimpleNamespace(chunked_prefill_size=-1),
            ),
            mock.patch.object(
                runtime_context,
                "get_exec",
                return_value=SimpleNamespace(features=features),
            ),
            mock.patch.dict(
                sys.modules, {"sglang.srt.managers.prefix_admission": checkpoint_module}
            ),
        ):
            # get_parallel itself is real. The unrelated numerical-backend gate
            # is isolated; unsupported topology must fail before invoking it.
            try:
                return self.runtime._model_identity(), checkpoint.call_count
            finally:
                self.checkpoint_calls = checkpoint.call_count

    def test_attention_dp_is_rejected_with_false_deprecated_flag(self):
        self.assertIs(resolved_view(self.args).enable_dp_attention, False)
        with self.assertRaisesRegex(CaptureContractError, "attention_DP"):
            self.identity(attn_dp_size=2, attn_tp_size=1)
        self.assertEqual(self.checkpoint_calls, 0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_ordinary_tp1_and_tp2_attention_dp1_keep_startup_identity(self):
        for width in (1, 2):
            with self.subTest(tp_size=width):
                (manifest, schema), checkpoint_calls = self.identity(
                    tp_size=width, attn_tp_size=width
                )
                self.assertEqual(manifest["tp_size"], width)
                self.assertEqual(
                    [feature.name for feature in schema.features],
                    ["aux", "last_hidden"],
                )
                self.assertEqual(checkpoint_calls, 1)
        self.assertFalse(torch.cuda.is_initialized())

    def test_replica_dp_pp_and_cp_startup_rejections_remain(self):
        for topology, diagnostic in (
            (dict(dp_size=2), "DP"),
            (dict(pp_size=2), "PP"),
            (dict(attn_cp_size=2, attn_tp_size=1), "CP"),
            (dict(dcp_size=2), "CP"),
        ):
            with self.subTest(topology=topology):
                with self.assertRaisesRegex(CaptureContractError, diagnostic):
                    self.identity(**topology)
                self.assertEqual(self.checkpoint_calls, 0)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
