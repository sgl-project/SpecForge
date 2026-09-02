import ast
import inspect
import textwrap
import types
import unittest
from unittest import mock

import torch

from specforge.offline_capture.sglang_backend import capture as sglang_capture
from specforge.offline_capture.sglang_backend import patch as sglang_patch
from specforge.offline_capture.sglang_backend import utils as sglang_utils


class SGLang0518CompatibilityTest(unittest.TestCase):
    def test_offline_requests_initialize_current_extend_range(self):
        tree = ast.parse(
            textwrap.dedent(
                inspect.getsource(
                    sglang_capture.OfflineSGLangCaptureBackend.capture_rows
                )
            )
        )
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "set_extend_range"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].args), 2)

    def test_tp_and_pdmux_calls_omit_removed_keywords(self):
        tree = ast.parse(
            textwrap.dedent(inspect.getsource(sglang_patch.initialize_model_parallel))
        )
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "init_model_parallel_group"
        ]
        by_group_name = {}
        for call in calls:
            keywords = {keyword.arg: keyword.value for keyword in call.keywords}
            group_name = keywords.get("group_name")
            if isinstance(group_name, ast.Constant):
                by_group_name[group_name.value] = set(keywords)

        for group_name in ("tp", "pdmux_prefill_tp"):
            self.assertIn(group_name, by_group_name)
            self.assertNotIn("pynccl_use_current_stream", by_group_name[group_name])
            self.assertNotIn("torch_compile", by_group_name[group_name])

    def test_model_runner_uses_parallel_state_and_current_mlp_sync_api(self):
        build_tree = ast.parse(
            textwrap.dedent(
                inspect.getsource(sglang_capture.OfflineSGLangCaptureBackend.build)
            )
        )
        runner_call = next(
            node
            for node in ast.walk(build_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "SGLangRunner"
        )
        runner_keywords = {keyword.arg for keyword in runner_call.keywords}
        self.assertIn("ps", runner_keywords)
        for removed_keyword in (
            "tp_rank",
            "tp_size",
            "moe_ep_rank",
            "moe_ep_size",
            "pp_rank",
            "pp_size",
        ):
            self.assertNotIn(removed_keyword, runner_keywords)

        mlp_tree = ast.parse(
            textwrap.dedent(
                inspect.getsource(
                    sglang_capture.OfflineSGLangCaptureBackend._maybe_prepare_mlp_sync_batch
                )
            )
        )
        mlp_call = next(
            node
            for node in ast.walk(mlp_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "prepare_mlp_sync_batch_raw"
        )
        self.assertIn("model_runner", {keyword.arg for keyword in mlp_call.keywords})

        forward_tree = ast.parse(
            textwrap.dedent(
                inspect.getsource(
                    sglang_capture.OfflineSGLangCaptureBackend._forward_extend
                )
            )
        )
        forward_call = next(
            node
            for node in ast.walk(forward_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "init_new"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ForwardBatch"
        )
        forward_keywords = {
            keyword.arg: keyword.value for keyword in forward_call.keywords
        }
        return_before_norm = forward_keywords.get("return_hidden_states_before_norm")
        self.assertIsInstance(return_before_norm, ast.Constant)
        self.assertFalse(return_before_norm.value)

    def test_dp_attention_rank_uses_current_runtime_flags(self):
        import sglang.srt.layers.dp_attention as dp_attention

        server_args = types.SimpleNamespace(
            enable_dp_attention=False,
            tp_size=2,
            dp_size=4,
            moe_dense_tp_size=None,
            pp_size=1,
            attn_cp_size=1,
            device="cpu",
        )
        model_config = types.SimpleNamespace(
            hidden_size=64,
            dtype=torch.float32,
            hf_config=types.SimpleNamespace(hybrid_override_pattern=None),
        )
        dp_flags = types.SimpleNamespace(enabled=None, max_len_with_idle=None)
        with (
            mock.patch.object(
                sglang_patch.parallel_state,
                "get_tensor_model_parallel_rank",
                return_value=1,
            ),
            mock.patch.object(
                sglang_patch,
                "compute_dp_attention_world_info",
                return_value=(11, 22, 33, 44),
            ),
            mock.patch.object(
                sglang_patch,
                "get_flags",
                return_value=types.SimpleNamespace(dp=dp_flags),
            ),
            mock.patch.object(sglang_patch._DpGatheredBufferWrapper, "set_metadata"),
            mock.patch.object(dp_attention, "_ATTN_DP_RANK", None, create=True),
            mock.patch.object(dp_attention, "_ATTN_DP_SIZE", None, create=True),
        ):
            sglang_patch.initialize_dp_attention(server_args, model_config)
            self.assertEqual(dp_attention._ATTN_DP_RANK, 33)
            self.assertEqual(dp_attention._ATTN_DP_SIZE, 1)
            self.assertFalse(dp_flags.enabled)
            self.assertFalse(dp_flags.max_len_with_idle)

    def test_multi_item_delimiter_is_read_from_forward_batch(self):
        class FakeForwardBatch:
            multi_item_delimiter_indices = [2, 5]

        metadata = types.SimpleNamespace(is_prefill_only=True)

        class FakeLogitsMetadata:
            @classmethod
            def from_forward_batch(cls, _batch):
                return metadata

        processor = mock.Mock()
        processor.compute_logprobs_for_multi_item_scoring.return_value = "sentinel"
        with (
            mock.patch.object(sglang_utils, "ForwardBatch", FakeForwardBatch),
            mock.patch.object(sglang_utils, "LogitsMetadata", FakeLogitsMetadata),
        ):
            result = sglang_utils.replaced_logits_processor_forward_for_offline_eagle3(
                processor,
                "input_ids",
                "hidden_states",
                "lm_head",
                FakeForwardBatch(),
            )

        self.assertEqual(result, "sentinel")
        processor.compute_logprobs_for_multi_item_scoring.assert_called_once_with(
            "input_ids",
            "hidden_states",
            "lm_head",
            metadata,
            [2, 5],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
