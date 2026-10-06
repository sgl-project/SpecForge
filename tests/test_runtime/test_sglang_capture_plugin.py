# coding=utf-8
"""Unit tests for the ``specforge-sglang-capture`` SGLang plugin (CPU, fakes).

The plugin's logic is driven with stand-ins for SGLang's batch, forward batch,
logits output and scheduler, so these run on any installed SGLang version.
``test_sglang_capture_plugin_integration.py`` drives the same plugin through
SGLang's real result-processing and streaming code when the installed SGLang
has the forward-observer extension points.
"""

import json
import os
import sys
import types
import unittest
from concurrent.futures import Future
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

import specforge_sglang_capture  # noqa: E402
from specforge_sglang_capture import capture  # noqa: E402


def _spec(sample_id, features=("aux", "last_hidden")):
    names = {"aux": "hidden_states", "last_hidden": "target"}
    return {
        "store_id": "store",
        "sample_id": sample_id,
        "gen": 1,
        "replace": False,
        "features": {artifact: names[artifact] for artifact in features},
        "passthrough": [],
    }


def _req(rid, prompt_len, spec=None, *, finished=True):
    custom_params = None if spec is None else {"spec_capture": json.dumps(spec)}
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=list(range(prompt_len)),
        sampling_params=SimpleNamespace(custom_params=custom_params),
        defer_output=False,
        finished=lambda: finished,
    )


def _forward_batch(extend_lens, prefix_lens=None, *, extend=True):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: extend),
        extend_seq_lens_cpu=list(extend_lens),
        extend_prefix_lens_cpu=list(prefix_lens or [0] * len(extend_lens)),
    )


def _logits_output(num_rows, *, aux_width=6, hidden=2, last=True, indices=None):
    aux = torch.arange(num_rows * aux_width, dtype=torch.float32).view(
        num_rows, aux_width
    )
    return SimpleNamespace(
        hidden_states=aux,
        last_hidden_states=(-aux[:, :hidden]).clone() if last else None,
        hidden_states_token_indices=indices,
    )


class FakeSink:
    def __init__(self):
        self.submissions = []

    def submit_samples(self, samples, *, ready_event=None):
        future = Future()
        self.submissions.append((samples, future))
        return future


def _runtime(*, is_writer=True, gpu_put=False, max_pending=2):
    return capture.CaptureRuntime(
        sink=FakeSink(),
        is_writer=is_writer,
        gpu_put=gpu_put,
        max_pending_batches=max_pending,
    )


def _host_copy(tensor):
    return tensor.clone()


class TestRequestSpec(unittest.TestCase):
    def test_parses_and_caches_the_json_spec(self):
        req = _req("r", 3, _spec("s"))
        spec = capture.request_spec(req)
        self.assertEqual(spec["sample_id"], "s")
        req.sampling_params.custom_params = None
        self.assertIs(capture.request_spec(req), spec)

    def test_requests_without_a_spec_are_not_captured(self):
        self.assertIsNone(capture.request_spec(_req("r", 3)))

    def test_malformed_spec_becomes_an_error_item(self):
        req = _req("r", 3)
        req.sampling_params.custom_params = {"spec_capture": "{not json"}
        items, _ = capture.collect_capture_items([req], [3], [0])
        self.assertIn("invalid spec_capture", items[0].error)


class TestCollectItems(unittest.TestCase):
    def test_offsets_follow_every_request_in_the_batch(self):
        reqs = [_req("a", 3, _spec("a")), _req("plain", 2), _req("b", 4, _spec("b"))]
        items, num_rows = capture.collect_capture_items(reqs, [3, 2, 4], [0, 0, 0])
        self.assertEqual(num_rows, 9)
        self.assertEqual(
            [(i.rid, i.start, i.length) for i in items],
            [
                ("a", 0, 3),
                ("b", 5, 4),
            ],
        )
        self.assertTrue(all(item.error is None for item in items))

    def test_cached_prefix_and_partial_prefill_are_errors(self):
        reqs = [_req("cached", 4, _spec("c")), _req("chunk", 8, _spec("k"))]
        items, _ = capture.collect_capture_items(reqs, [2, 4], [2, 0])
        self.assertIn("prefix cache", items[0].error)
        self.assertIn("--chunked-prefill-size -1", items[1].error)


class TestObserver(unittest.TestCase):
    def test_non_writer_ranks_and_decode_batches_capture_nothing(self):
        batch = SimpleNamespace(reqs=[_req("a", 2, _spec("a"))])
        out = _logits_output(2)
        observer = _runtime(is_writer=False).observer
        self.assertIsNone(
            observer.after_forward(batch, _forward_batch([2]), out, can_run_graph=False)
        )
        observer = _runtime().observer
        self.assertIsNone(
            observer.after_forward(
                batch, _forward_batch([1], extend=False), out, can_run_graph=False
            )
        )

    def test_batches_without_capture_requests_produce_no_output(self):
        batch = SimpleNamespace(reqs=[_req("plain", 2)])
        output = _runtime().observer.after_forward(
            batch, _forward_batch([2]), _logits_output(2), can_run_graph=False
        )
        self.assertIsNone(output)

    def test_capture_only_batch_keeps_whole_tensors(self):
        batch = SimpleNamespace(
            reqs=[_req("a", 2, _spec("a")), _req("b", 3, _spec("b"))]
        )
        out = _logits_output(5)
        device = _runtime().observer.after_forward(
            batch, _forward_batch([2, 3]), out, can_run_graph=False
        )
        self.assertIs(device.aux, out.hidden_states)
        self.assertIs(device.last_hidden, out.last_hidden_states)

    def test_graph_outputs_are_copied_out_of_static_buffers(self):
        batch = SimpleNamespace(reqs=[_req("a", 2, _spec("a"))])
        out = _logits_output(2)
        device = _runtime().observer.after_forward(
            batch, _forward_batch([2]), out, can_run_graph=True
        )
        self.assertIsNot(device.aux, out.hidden_states)
        torch.testing.assert_close(device.aux, out.hidden_states)

    def test_mixed_batch_gathers_capture_rows_and_rebases_items(self):
        reqs = [_req("plain", 2), _req("a", 3, _spec("a")), _req("b", 1, _spec("b"))]
        out = _logits_output(6)
        device = _runtime().observer.after_forward(
            SimpleNamespace(reqs=reqs),
            _forward_batch([2, 3, 1]),
            out,
            can_run_graph=False,
        )
        torch.testing.assert_close(device.aux, out.hidden_states[2:6])
        self.assertEqual([(i.rid, i.start) for i in device.items], [("a", 0), ("b", 3)])

    def test_missing_last_hidden_fails_only_requests_that_ask_for_it(self):
        reqs = [
            _req("both", 2, _spec("both")),
            _req("aux", 2, _spec("aux", features=("aux",))),
        ]
        device = _runtime().observer.after_forward(
            SimpleNamespace(reqs=reqs),
            _forward_batch([2, 2]),
            _logits_output(4, last=False),
            can_run_graph=False,
        )
        both, aux_only = device.items
        self.assertIn("last_hidden", both.error)
        self.assertIsNone(aux_only.error)

    def test_mup_targets_restore_the_pre_head_scale_last_hidden(self):
        runtime = _runtime()
        runtime.last_hidden_scale = 24.0
        out = _logits_output(2)
        device = runtime.observer.after_forward(
            SimpleNamespace(reqs=[_req("a", 2, _spec("a"))]),
            _forward_batch([2]),
            out,
            can_run_graph=False,
        )
        torch.testing.assert_close(device.last_hidden, out.last_hidden_states * 24.0)
        self.assertIs(device.aux, out.hidden_states)

    def test_partial_row_outputs_fail_every_capture_request(self):
        device = _runtime().observer.after_forward(
            SimpleNamespace(reqs=[_req("a", 2, _spec("a"))]),
            _forward_batch([2]),
            _logits_output(2, indices=torch.tensor([1])),
            can_run_graph=False,
        )
        self.assertIn("subset", device.items[0].error)


class TestConsume(unittest.TestCase):
    def _consume(self, runtime, reqs, extend_lens, out):
        batch = SimpleNamespace(reqs=reqs)
        device = runtime.observer.after_forward(
            batch, _forward_batch(extend_lens), out, can_run_graph=False
        )
        host = device.copy_to_host(_host_copy)
        host.consume(batch, commits=None)
        return host

    def test_finished_requests_are_published_and_held(self):
        runtime = _runtime()
        reqs = [_req("a", 2, _spec("a")), _req("b", 3, _spec("b"))]
        out = _logits_output(5)
        self._consume(runtime, reqs, [2, 3], out)

        [(samples, _future)] = runtime.sink.submissions
        self.assertEqual([spec["sample_id"] for spec, _, _ in samples], ["a", "b"])
        torch.testing.assert_close(samples[1][1], out.hidden_states[2:5])
        torch.testing.assert_close(samples[1][2], out.last_hidden_states[2:5])
        self.assertTrue(all(req.defer_output for req in reqs))
        self.assertTrue(runtime.pending.has_pending())

    def test_errors_stream_immediately_with_an_error_result(self):
        runtime = _runtime()
        req = _req("a", 2, _spec("a"), finished=False)
        self._consume(runtime, [req], [2], _logits_output(2))

        self.assertEqual(runtime.sink.submissions, [])
        self.assertFalse(req.defer_output)
        self.assertIn("max_new_tokens=0", runtime.results["a"]["error"])
        self.assertEqual(runtime.results["a"]["sample_id"], "a")


class TestPendingCaptures(unittest.TestCase):
    def _submit(self, runtime, rid):
        req = _req(rid, 1, _spec(rid))
        runtime.pending.submit([req], [(_spec(rid), None, None)])
        return req, runtime.sink.submissions[-1][1]

    def test_releases_in_submission_order(self):
        runtime = _runtime(max_pending=4)
        first, first_future = self._submit(runtime, "a")
        second, second_future = self._submit(runtime, "b")

        second_future.set_result([{"sample_id": "b"}])
        self.assertEqual(runtime.pending.poll(), [])

        first_future.set_result([{"sample_id": "a"}])
        self.assertEqual(runtime.pending.poll(), [first, second])
        self.assertEqual(
            runtime.results, {"a": {"sample_id": "a"}, "b": {"sample_id": "b"}}
        )
        self.assertFalse(runtime.pending.has_pending())

    def test_a_failed_batch_releases_its_requests_with_error_results(self):
        runtime = _runtime()
        req, future = self._submit(runtime, "a")
        future.set_exception(RuntimeError("put failed"))

        self.assertEqual(runtime.pending.poll(), [req])
        self.assertEqual(
            runtime.results["a"], {"sample_id": "a", "error": "put failed"}
        )

    def test_backpressure_waits_for_the_oldest_batch(self):
        runtime = _runtime(max_pending=1)
        _, first_future = self._submit(runtime, "a")
        waited = []
        with mock.patch.object(
            capture, "wait_futures", side_effect=lambda fs: waited.extend(fs)
        ):
            self._submit(runtime, "b")
        self.assertEqual(waited, [first_future])


class FakeStreamer:
    has_additional_customized_info = False

    def should_build_additional_customized_info(self):
        return True

    def build_additional_customized_info(self, reqs):
        return {}


class FakeBuildingStreamer(FakeStreamer):
    has_additional_customized_info = True

    def build_additional_customized_info(self, reqs):
        return {"other": [["x"] for _ in reqs]}


class TestStreamer(unittest.TestCase):
    def test_streamer_returns_each_result_once(self):
        streamer = capture.capture_streamer_class(FakeStreamer)()
        self.assertTrue(streamer.has_additional_customized_info)
        self.assertFalse(streamer.should_build_additional_customized_info())

        streamer.spec_capture_results = {"a": {"sample_id": "a"}}
        reqs = [SimpleNamespace(rid="a"), SimpleNamespace(rid="plain")]
        self.assertTrue(streamer.should_build_additional_customized_info())
        self.assertEqual(
            streamer.build_additional_customized_info(reqs),
            {"spec_capture": [[{"sample_id": "a"}], []]},
        )
        self.assertFalse(streamer.should_build_additional_customized_info())

    def test_streamer_keeps_fields_of_the_class_it_extends(self):
        streamer = capture.capture_streamer_class(FakeBuildingStreamer)()
        streamer.spec_capture_results = {"a": {"sample_id": "a"}}
        info = streamer.build_additional_customized_info([SimpleNamespace(rid="a")])
        self.assertEqual(
            info, {"other": [["x"]], "spec_capture": [[{"sample_id": "a"}]]}
        )


def _fake_sglang_modules(features, *, chunked_prefill_size=-1, attn_tp_rank=0):
    runtime_context = types.ModuleType("sglang.srt.runtime_context")
    runtime_context.get_exec = lambda: SimpleNamespace(features=features)
    runtime_context.get_parallel = lambda: SimpleNamespace(attn_tp_rank=attn_tp_rank)
    runtime_context.get_schedule = lambda: SimpleNamespace(
        chunked_prefill_size=chunked_prefill_size
    )
    return {"sglang.srt.runtime_context": runtime_context}


class _ModelRunner:
    forward_observer = None


def _scheduler():
    scheduler = SimpleNamespace(
        tp_worker=SimpleNamespace(model_runner=_ModelRunner()),
        output_streamer=SimpleNamespace(),
        sources=[],
    )
    scheduler.register_deferred_output_source = scheduler.sources.append
    return scheduler


class TestInstall(unittest.TestCase):
    FEATURES = dict(
        aux_hidden_state_capture="dflash",
        aux_hidden_state_layer_ids=[1, 3],
        return_hidden_states_mode="full",
    )

    def _install(self, scheduler=None, **overrides):
        fields = {**self.FEATURES, **overrides}
        chunked = fields.pop("chunked_prefill_size", -1)
        modules = _fake_sglang_modules(
            SimpleNamespace(**fields), chunked_prefill_size=chunked
        )
        scheduler = scheduler or _scheduler()
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.dict(os.environ, {"MOONCAKE_PROTOCOL": "tcp"}),
        ):
            return scheduler, capture.install(scheduler)

    def test_wires_observer_source_and_results(self):
        scheduler, runtime = self._install()
        self.assertIs(
            scheduler.tp_worker.model_runner.forward_observer, runtime.observer
        )
        self.assertEqual(scheduler.sources, [runtime.pending])
        self.assertIs(scheduler.output_streamer.spec_capture_results, runtime.results)
        self.assertEqual(runtime.sink.aux_layer_ids, [1, 3])
        self.assertTrue(runtime.is_writer)
        self.assertFalse(runtime.gpu_put)

    def test_reads_the_mup_multiplier_from_the_target_config(self):
        scheduler = _scheduler()
        scheduler.tp_worker.model_runner.model_config = SimpleNamespace(
            hf_text_config=SimpleNamespace(logits_mup_width_multiplier=24),
            hf_config=SimpleNamespace(),
        )
        _, runtime = self._install(scheduler)
        self.assertEqual(runtime.last_hidden_scale, 24.0)
        _, runtime = self._install()
        self.assertIsNone(runtime.last_hidden_scale)

    def test_rejects_servers_missing_required_flags(self):
        for overrides, match in (
            (dict(aux_hidden_state_capture=None), "--aux-hidden-state-capture"),
            (dict(return_hidden_states_mode=None), "--return-hidden-states-mode full"),
            (dict(chunked_prefill_size=8192), "--chunked-prefill-size -1"),
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(RuntimeError, match):
                    self._install(**overrides)

    def test_rejects_sglang_builds_without_the_extension_points(self):
        scheduler = _scheduler()
        scheduler.tp_worker.model_runner = SimpleNamespace()
        with self.assertRaisesRegex(RuntimeError, "forward observers"):
            self._install(scheduler)


class TestRegister(unittest.TestCase):
    def _fake_hook_registry(self):
        registered = []
        module = types.ModuleType("sglang.srt.plugins.hook_registry")
        module.HookType = SimpleNamespace(AFTER="after")
        module.HookRegistry = SimpleNamespace(
            register=lambda target, hook, hook_type: registered.append(
                (target, hook_type)
            )
        )
        return registered, {"sglang.srt.plugins.hook_registry": module}

    def test_disabled_plugin_registers_nothing(self):
        registered, modules = self._fake_hook_registry()
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.dict(os.environ, {"SPECFORGE_SPEC_CAPTURE": "0"}),
        ):
            specforge_sglang_capture.register()
        self.assertEqual(registered, [])

    def test_enabled_plugin_hooks_the_scheduler(self):
        registered, modules = self._fake_hook_registry()
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.dict(os.environ, {"SPECFORGE_SPEC_CAPTURE": "1"}),
        ):
            specforge_sglang_capture.register()
        self.assertEqual(
            registered,
            [
                (
                    "sglang.srt.managers.scheduler.Scheduler.get_output_streamer_class",
                    "after",
                ),
                ("sglang.srt.managers.scheduler.Scheduler.__init__", "after"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
