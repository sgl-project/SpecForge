import ast
import inspect
import textwrap
import types
import unittest
from unittest import mock

import torch

from specforge.offline_capture.sglang_backend import patch as sglang_patch
from specforge.offline_capture.sglang_backend import utils as sglang_utils


class SGLang0514CompatibilityTest(unittest.TestCase):
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

    def test_dp_attention_rank_uses_third_world_info_item(self):
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
        model_config = types.SimpleNamespace(hidden_size=64, dtype=torch.float32)
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
                "compute_dp_attention_local_info",
                return_value=(55, 66, 77),
            ),
            mock.patch.object(sglang_patch._DpGatheredBufferWrapper, "set_metadata"),
            mock.patch.object(dp_attention, "_ATTN_DP_RANK", None, create=True),
            mock.patch.object(dp_attention, "_LOCAL_ATTN_DP_RANK", None, create=True),
        ):
            sglang_patch.initialize_dp_attention(server_args, model_config)
            self.assertEqual(dp_attention._ATTN_DP_RANK, 33)
            self.assertEqual(dp_attention._LOCAL_ATTN_DP_RANK, 77)

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
