# coding=utf-8
"""The capture plugin driven through SGLang's real result and streaming code.

Skipped unless the installed SGLang has the forward-observer, deferred-output
and aux-capture extension points. On such a build this pins the whole
scheduler-side path without a GPU: observer output -> the generation result's
host copy -> ``HostAuxiliaryOutput.consume`` -> held response -> the real
``SpecCaptureSink`` writing raw bytes into an in-memory Mooncake stand-in ->
deferred release -> the streamer's customized output -> the tokenizer
manager's ``meta_info`` -> the SpecForge adapter's result parsing.
"""

import ctypes
import importlib.util
import json
import os
import sys
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        "plugins",
        "sglang-spec-capture",
    ),
)


def _has_extension_points() -> bool:
    try:
        if (
            importlib.util.find_spec("sglang.srt.model_executor.forward_observer")
            is None
        ):
            return False
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    except Exception:
        return False
    return "last_hidden_states" in LogitsProcessorOutput.__dataclass_fields__


class FakeMooncakeStore:
    """Records ``batch_put_from`` writes as raw bytes keyed by object key."""

    def __init__(self):
        self.objects = {}
        self.registered = set()

    def register_buffer(self, ptr, nbytes):
        self.registered.add(ptr)
        return 0

    def unregister_buffer(self, ptr):
        self.registered.discard(ptr)
        return 0

    def batch_put_from(self, keys, ptrs, sizes, config):
        for key, ptr, size in zip(keys, ptrs, sizes):
            self.objects[key] = ctypes.string_at(ptr, size)
        return [0] * len(keys)

    def remove(self, key):
        self.objects.pop(key, None)


class _Req:
    """The request fields SGLang's streamer and the plugin read."""

    def __init__(self, rid, prompt_len, spec=None):
        self.rid = rid
        self.origin_input_ids = list(range(prompt_len))
        custom_params = None if spec is None else {"spec_capture": json.dumps(spec)}
        self.sampling_params = SimpleNamespace(
            custom_params=custom_params,
            stream_interval=None,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
            no_stop_trim=False,
        )
        self.defer_output = False
        self.http_worker_ipc = None
        self.finished_reason = SimpleNamespace(to_json=lambda: {"type": "length"})
        self.finished_output = False
        self.finished_len = None
        self.beam_group = None
        self.stream = False
        self.output_ids = [0]
        self.output_ids_through_stop = self.output_ids
        self.send_token_offset = 0
        self.send_output_token_logprobs_offset = 0
        self.send_decode_id_offset = 0
        self.decoded_text = ""
        self.reasoning_tokens = 0
        self.cached_tokens = 0
        self.cached_tokens_device = 0
        self.cached_tokens_host = 0
        self.cached_tokens_storage = 0
        self.retraction_count = 0
        self.time_stats = None
        self.return_hidden_states = False
        self.return_routed_experts = False
        self.return_indexer_topk = False
        self.return_sampling_mask = False
        self.return_logprob = False
        self.mm_image_tokens = 0
        self.mm_audio_tokens = 0
        self.mm_video_tokens = 0
        self.multimodal_inputs = None
        self.customized_info = None
        self.weight_version_events = []

    def finished(self):
        return True

    def init_incremental_detokenize(self):
        return self.output_ids_through_stop, 0

    def check_match_stop_str_prefix(self):
        return False


def _spec(sample_id):
    return {
        "store_id": "store",
        "sample_id": sample_id,
        "gen": 1,
        "replace": False,
        "features": {"aux": "hidden_states", "last_hidden": "target"},
        "passthrough": [
            {"name": "input_ids", "data": [7, 8, 9], "shape": [1, 3], "dtype": "int64"}
        ],
    }


class _CopyDone:
    def record(self):
        pass

    def synchronize(self):
        pass


@unittest.skipUnless(
    _has_extension_points(),
    "installed SGLang lacks the forward-observer extension points",
)
class TestCapturePluginOnSGLang(unittest.TestCase):
    def setUp(self):
        from sglang.test.test_utils import enter_scope, published_topology

        for name, value in (
            ("get_serving", SimpleNamespace(stream_interval=1, weight_version="w")),
            (
                "get_observability",
                SimpleNamespace(enable_request_time_stats_logging=False),
            ),
        ):
            patcher = mock.patch(
                f"sglang.srt.managers.scheduler_components.output_streamer.{name}",
                return_value=value,
            )
            patcher.start()
            self.addCleanup(patcher.stop)
        enter_scope(self, published_topology(ranks={"dp_rank": 0}))

    def _scheduler(self, outputs, runtime):
        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.managers.scheduler_components.output_streamer import (
            SchedulerOutputStreamer,
        )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from specforge_sglang_capture.capture import capture_streamer_class

        streamer = capture_streamer_class(SchedulerOutputStreamer)(
            send_to_detokenizer=SimpleNamespace(send_output=outputs.append),
            tree_cache=None,
            server_args=SimpleNamespace(enable_lmcache=False),
            is_generation=True,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            disaggregation_mode=DisaggregationMode.NULL,
            enable_hicache_storage=lambda: False,
        )
        scheduler = object.__new__(Scheduler)
        scheduler.output_streamer = streamer
        scheduler.register_deferred_output_source(runtime.pending)
        streamer.spec_capture_results = runtime.results
        return scheduler

    def test_capture_is_published_then_returned_in_meta_info(self):
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.managers.io_struct import unwrap_from_pickle
        from sglang.srt.managers.scheduler_components.batch_result_processor import (
            SchedulerBatchResultProcessor,
        )
        from sglang.srt.managers.tokenizer_manager import TokenizerManager
        from sglang.srt.managers.utils import GenerationBatchResult
        from specforge_sglang_capture.capture import CaptureRuntime
        from specforge_sglang_capture.sink import SpecCaptureSink

        from specforge.inference.adapters.server_capture import _capture_result_for_task

        store = FakeMooncakeStore()
        sink = SpecCaptureSink(aux_layer_ids=[1, 3])
        sink._store, sink._put_config = store, object()
        runtime = CaptureRuntime(
            sink=sink, is_writer=True, gpu_put=False, max_pending_batches=2
        )
        outputs = []
        scheduler = self._scheduler(outputs, runtime)

        reqs = [_Req("plain", 2), _Req("cap", 3, _spec("cap"))]
        batch = SimpleNamespace(reqs=reqs)
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: True),
            extend_seq_lens_cpu=[2, 3],
            extend_prefix_lens_cpu=[0, 0],
        )
        aux = torch.randn(5, 6, dtype=torch.bfloat16)
        last_hidden = torch.randn(5, 2, dtype=torch.bfloat16)
        logits_output = LogitsProcessorOutput(
            next_token_logits=None, hidden_states=aux, last_hidden_states=last_hidden
        )
        result = GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=torch.tensor([0, 0]),
            copy_done=_CopyDone(),
            forward_auxiliary_output=runtime.observer.after_forward(
                batch, forward_batch, logits_output, can_run_graph=False
            ),
        )

        result.copy_to_cpu(return_logprob=False)
        SchedulerBatchResultProcessor.consume_auxiliary_output(
            batch, result.auxiliary_host_output, [0, 0]
        )
        scheduler.output_streamer.stream_output(batch.reqs, False)

        # Only the plain request streams while the capture is being written.
        self.assertEqual([o.rids for o in outputs], [["plain"]])
        self.assertTrue(scheduler.has_pending_deferred_outputs())

        deadline = time.monotonic() + 30
        while not outputs[1:] and time.monotonic() < deadline:
            scheduler.stream_released_deferred_outputs()
            time.sleep(0.01)
        self.assertEqual(outputs[1].rids, ["cap"])
        self.assertFalse(scheduler.has_pending_deferred_outputs())

        # The objects hold the capture request's rows, byte for byte.
        def stored(name, like):
            raw = store.objects[f"store/cap/g1/{name}"]
            return torch.frombuffer(bytearray(raw), dtype=like.dtype).view(
                1, *like.shape
            )

        torch.testing.assert_close(stored("hidden_states", aux[2:5]), aux[2:5][None])
        torch.testing.assert_close(
            stored("target", last_hidden[2:5]), last_hidden[2:5][None]
        )
        self.assertEqual(
            torch.frombuffer(
                bytearray(store.objects["store/cap/g1/input_ids"]), dtype=torch.int64
            ).tolist(),
            [7, 8, 9],
        )
        self.assertEqual(store.registered, set())

        # The response metadata the producer parses.
        meta_info = {}
        manager = object.__new__(TokenizerManager)
        state = SimpleNamespace(customized_info_accumulated={})
        manager.update_request_meta_info(
            meta_info,
            state,
            unwrap_from_pickle(outputs[1].customized_info),
            0,
            None,
        )
        capture_result = _capture_result_for_task(
            meta_info["spec_capture"], task_id="cap", expected_sample_id="cap"
        )
        self.assertEqual(capture_result["store_id"], "store")
        self.assertEqual(capture_result["aux_layer_ids"], [1, 3])
        self.assertEqual(
            capture_result["features"],
            {
                "hidden_states": {"shape": [1, 3, 6], "dtype": "bfloat16"},
                "target": {"shape": [1, 3, 2], "dtype": "bfloat16"},
                "input_ids": {"shape": [1, 3], "dtype": "int64"},
            },
        )


if __name__ == "__main__":
    unittest.main()
