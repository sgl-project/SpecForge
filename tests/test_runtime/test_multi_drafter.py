# coding=utf-8
"""Several drafters trained from one capture stack (no GPU, server or Mooncake).

The real capture plugin sink stands in for the server: it cuts each drafter's
layers from one capture and writes them into the fake Mooncake backend that the
real client store reads back.
"""

import dataclasses
import json
import os
import sys
import tempfile
import threading
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

from specforge_sglang_capture import _capture_internal_state  # noqa: E402
from specforge_sglang_capture import sink as plugin_sink  # noqa: E402

from specforge.application import resolve_run
from specforge.cli import _config_for_role
from specforge.config import Config, flatten_drafter
from specforge.data.loss_mask import has_consecutive_supervised_tokens
from specforge.inference.adapters.server_capture import (
    CaptureOutput,
    ServerCaptureFailure,
    ServerCaptureSchema,
    SGLangServerCaptureAdapter,
)
from specforge.inference.capture import CaptureConfig
from specforge.launch import ProducerDrafter, _DrafterLane, build_disagg_online_producer
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel
from specforge.training.assembly import _prepare_prompts, _prompt_cache_key
from specforge.training.capture_contract import (
    ServerCaptureContract,
    combine_capture_contracts,
)
from specforge.training.disaggregated import (
    _ONLINE_SCHEDULE_SUFFIX,
    _build_online,
    build_disaggregated_run,
)
from tests.test_algorithms.fixtures import builtin_algorithm_registry
from tests.test_runtime.test_server_capture import HIDDEN, _FakeMooncakeStore, _task

REGISTRY = builtin_algorithm_registry()
PRESET = "specforge/config/draft/presets/qwen3-8b-{}.yaml"
#: Each drafter's capture layers; the servers capture their sorted union.
LAYERS = {"dflash": (2, 8), "dspark": (5, 8)}
SERVER_LAYERS = (2, 5, 8)
URL = "http://server:30000"


class _PluginServer:
    """``post_fn`` served by the real plugin sink, writing into *backend*."""

    def __init__(self, backend, layers=SERVER_LAYERS):
        self.sink = plugin_sink.SpecCaptureSink(aux_layer_ids=list(layers))
        self.sink._store = backend
        self.raw_specs = []
        #: Request sample id -> the (layer rows, final hidden rows) captured.
        self.captured = {}
        #: Called with the number of each request before it is served.
        self.on_request = lambda number: None
        self._lock = threading.Lock()

    @property
    def specs(self):
        return [json.loads(raw) for raw in self.raw_specs]

    def server_info(self, url, timeout):
        runtime = SimpleNamespace(is_writer=False, sink=self.sink, capture_method="x")
        state = SimpleNamespace(internal_state={})
        scheduler = SimpleNamespace(_specforge_capture_runtime=runtime)
        _capture_internal_state(state, scheduler)
        return {"internal_states": [state.internal_state]}

    def __call__(self, url, json_body, timeout):
        rows = []
        for ids, params in zip(json_body["input_ids"], json_body["sampling_params"]):
            raw = params["custom_params"]["spec_capture"]
            spec = json.loads(raw)
            aux = torch.randn(len(ids), len(self.sink.aux_layer_ids) * HIDDEN)
            last = torch.randn(len(ids), HIDDEN)
            with self._lock:
                self.raw_specs.append(raw)
                self.on_request(len(self.raw_specs))
                self.captured[spec["sample_id"]] = (aux, last)
                (result,) = self.sink.put_samples([(spec, aux, last)])
            rows.append({"meta_info": {"spec_capture": [result]}})
        return rows


def _capture(name):
    algorithm = REGISTRY.resolve(name)
    return CaptureConfig.from_strategy(
        required_features=algorithm.spec.feature_contract(
            "streaming", "text"
        ).required_tensors,
        aux_hidden_state_layer_ids=LAYERS[name],
        target_repr=algorithm.providers.server_streaming_for(
            "text"
        ).target_representation,
        target_hidden_size=HIDDEN,
    )


def _output(name, **kwargs):
    layout = REGISTRY.resolve(name).providers.server_streaming_for("text").layout
    return CaptureOutput(
        run_id=f"run-{name}",
        algorithm=name,
        schema=ServerCaptureSchema(
            aux_feature=layout.aux_feature,
            last_hidden_feature=layout.last_hidden_feature,
            passthrough=layout.passthrough,
            attention_mask_feature=layout.attention_mask_feature,
        ),
        **kwargs,
    )


def _prompts(count):
    return [
        {"task_id": f"p{i}", "payload": {"input_ids": [1, 2, 3], "loss_mask": [1] * 3}}
        for i in range(count)
    ]


def _samples(backend, name):
    """Sample ids of one drafter that still hold objects in the store."""
    return {key.split("/")[1] for key in backend._d if f"run-{name}:" in key}


class _MultiDrafterCase(unittest.TestCase):
    def setUp(self):
        # The sink publishes from host memory through the fake store.
        self.enterContext(
            mock.patch.dict(
                os.environ,
                {"MOONCAKE_PROTOCOL": "tcp", "SGLANG_SPEC_CAPTURE_GPU_PUT": "0"},
            )
        )
        self.backend = _FakeMooncakeStore()
        self.server = _PluginServer(self.backend)
        self.addCleanup(self.server.sink._executor.shutdown)
        self.store = MooncakeFeatureStore(store=self.backend, store_id="run")

    def _adapter(self, outputs=None, **kwargs):
        kwargs.setdefault("server_info_fn", self.server.server_info)
        return SGLangServerCaptureAdapter(
            URL,
            self.store,
            run_id="run",
            outputs=outputs or [_output("dflash"), _output("dspark")],
            post_fn=self.server,
            **kwargs,
        )


class TestCaptureForSeveralDrafters(_MultiDrafterCase):
    captures = [_capture("dflash"), _capture("dspark")]

    def test_each_drafter_reads_its_own_layers_of_one_capture(self):
        tasks = [_task(0, 4), _task(1, 6)]
        results = self._adapter().produce_refs(tasks, capture=self.captures)

        self.assertEqual(
            [spec["sample_id"] for spec in self.server.specs], ["run:t0", "run:t1"]
        )
        for task, (dflash, dspark) in zip(tasks, results):
            aux, last = self.server.captured[f"run:{task.task_id}"]
            self.assertEqual(
                (dflash.sample_id, dflash.run_id, dflash.strategy),
                (f"run-dflash:{task.task_id}", "run-dflash", "dflash"),
            )
            self.assertEqual((dspark.run_id, dspark.strategy), ("run-dspark", "dspark"))
            tensors, _ = self.store.get(dflash)
            # No final hidden state for the drafter that did not ask for it.
            self.assertEqual(set(tensors), {"hidden_states", "input_ids", "loss_mask"})
            torch.testing.assert_close(
                tensors["hidden_states"][0],
                torch.cat([aux[:, :HIDDEN], aux[:, 2 * HIDDEN :]], dim=1),
            )
            tensors, _ = self.store.get(dspark)
            torch.testing.assert_close(tensors["hidden_states"][0], aux[:, HIDDEN:])
            torch.testing.assert_close(tensors["target_last_hidden_states"][0], last)

    def test_one_drafter_sends_todays_request(self):
        self.server.sink.aux_layer_ids = list(LAYERS["dflash"])
        output = _output("dflash")
        no_info = mock.Mock(side_effect=AssertionError("/server_info was read"))
        adapters = [
            SGLangServerCaptureAdapter(
                URL,
                self.store,
                run_id=output.run_id,
                algorithm=output.algorithm,
                schema=output.schema,
                post_fn=self.server,
                server_info_fn=no_info,
            ),
            self._adapter([output], server_info_fn=no_info),
        ]
        for adapter in adapters:
            (ref,) = adapter.produce_refs([_task(0, 4)], capture=_capture("dflash"))
            self.assertEqual(ref.sample_id, "run-dflash:t0")

        def passthrough(name, data):
            return {"name": name, "data": data, "shape": [1, 4], "dtype": "int64"}

        request = {
            "store_id": "run",
            "sample_id": "run-dflash:t0",
            "gen": 1,
            "replace": False,
            "features": {"aux": "hidden_states"},
            "passthrough": [
                passthrough("input_ids", [1, 2, 3, 4]),
                passthrough("loss_mask", [0, 0, 1, 1]),
            ],
        }
        self.assertEqual(
            self.server.raw_specs, [json.dumps(request, separators=(",", ":"))] * 2
        )

    def test_a_bad_or_lost_output_frees_every_output_of_the_prompt(self):
        adapter = self._adapter()
        wrong_width = dataclasses.replace(
            _capture("dspark"), target_hidden_size=HIDDEN + 1
        )
        (failure,) = adapter.produce_refs(
            [_task(0, 4)], capture=[_capture("dflash"), wrong_width]
        )
        self.assertIsInstance(failure, ServerCaptureFailure)
        self.assertFalse(failure.retryable)
        self.assertEqual(self.backend._d, {})
        self.assertEqual(self.store.discard_external_attempts(), 0)

        def lose_the_response(*args, **kwargs):
            self.server(*args, **kwargs)
            raise TimeoutError("response lost")

        adapter.post_fn = lose_the_response
        with self.assertRaisesRegex(TimeoutError, "response lost"):
            adapter.produce_refs([_task(1, 4)], capture=self.captures)
        self.assertEqual(_samples(self.backend, "dflash"), {"run-dflash:t1"})
        self.assertEqual(_samples(self.backend, "dspark"), {"run-dspark:t1"})
        self.assertEqual(self.store.discard_external_attempts(), 2)
        self.assertEqual(self.backend._d, {})

    def test_a_server_that_cannot_serve_the_outputs_gets_no_request(self):
        plugin = {"spec_capture_backend": "plugin"}
        cases = (
            (plugin, "cannot write several capture outputs"),
            (
                {**plugin, "spec_capture_outputs": True},
                r"captures layers \[\], but the drafters need \[2, 5, 8\]",
            ),
            (
                {
                    **plugin,
                    "spec_capture_outputs": True,
                    "spec_capture_aux_layer_ids": [8, 5, 2],
                },
                "in ascending order",
            ),
        )
        for state, message in cases:
            with self.subTest(state=state):
                adapter = self._adapter(
                    server_info_fn=lambda url, timeout, state=state: {
                        "internal_states": [state]
                    }
                )
                with self.assertRaisesRegex(RuntimeError, message):
                    adapter.produce_refs([_task(0, 4)], capture=self.captures)
                self.assertEqual(self.server.raw_specs, [])

    def test_servers_capture_the_sorted_union_under_one_method(self):
        def contract(method, layers):
            return ServerCaptureContract(method, layers, HIDDEN, 32, 32)

        dflash, eagle3 = contract("dflash", (1, 9, 17)), contract("eagle3", (1, 16))
        self.assertIs(combine_capture_contracts([eagle3]), eagle3)
        with self.assertLogs("specforge.training.capture_contract", "WARNING"):
            combined = combine_capture_contracts([eagle3, dflash])
        self.assertEqual(
            (combined.method, combined.aux_layer_ids), ("dflash", (1, 9, 16, 17))
        )
        with self.assertRaisesRegex(ValueError, "ascending target layer ids"):
            combine_capture_contracts([dflash, contract("dflash", (9, 1))])


class TestProducerForSeveralDrafters(_MultiDrafterCase):
    def _channels(self, dflash_channel=StreamingRefChannel):
        root = tempfile.mkdtemp(prefix="multi_drafter_")
        return {
            "dflash": dflash_channel(os.path.join(root, "dflash", "refs.jsonl")),
            "dspark": StreamingRefChannel(os.path.join(root, "dspark", "refs.jsonl")),
        }

    def _producer(self, channels, prompts, **kwargs):
        for channel in channels.values():
            channel.publish_consumer_quantum(1)
        outputs = [
            _output(name, wanted=lambda channel=channel: not channel.consumer_stopped())
            for name, channel in channels.items()
        ]
        _, drive = build_disagg_online_producer(
            drafters=[
                ProducerDrafter(
                    algorithm=REGISTRY.resolve(name),
                    channel=channel,
                    run_id=f"run-{name}",
                    aux_hidden_state_layer_ids=LAYERS[name],
                )
                for name, channel in channels.items()
            ],
            feature_source=self._adapter(outputs),
            prompts=prompts,
            feature_store=self.store,
            run_id="run",
            target_hidden_size=HIDDEN,
            lease=1,
            sleep=lambda _seconds: time.sleep(0.001),
            **kwargs,
        )
        return drive

    def _start(self, drive, channels):
        """Run *drive* until every trainer has stopped or one has failed."""
        outcome = {}

        def stopped():
            return all(c.consumer_stopped() for c in channels.values()) or any(
                c.consumer_failure() is not None for c in channels.values()
            )

        def run():
            try:
                outcome["produced"] = drive(should_stop=stopped)
            except BaseException as exc:
                outcome["error"] = exc

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread, outcome

    def _wait(self, condition, *, train=None, timeout=10):
        """Poll *condition*, letting the *train* reader consume all it is sent."""
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            refs = train.poll() if train is not None else None
            if refs:
                train.mark_consumed(len(refs))
            time.sleep(0.001)
        return condition()

    def test_every_channel_gets_the_prompts_in_the_same_order(self):
        channels = self._channels()
        drive = self._producer(channels, _prompts(6))

        self.assertEqual(drive(), 6)

        refs = {
            name: StreamingRefChannel(channel.path).poll()
            for name, channel in channels.items()
        }
        order = [ref.source_task_id for ref in refs["dflash"]]
        self.assertEqual(sorted(order), [f"p{i}" for i in range(6)])
        self.assertEqual([ref.source_task_id for ref in refs["dspark"]], order)
        for name, channel in channels.items():
            self.assertEqual({ref.run_id for ref in refs[name]}, {f"run-{name}"})
            self.assertTrue(channel.is_closed())

    def test_the_fullest_channel_pauses_capture_until_its_drafter_stops(self):
        channels = self._channels()
        drive = self._producer(
            channels,
            _prompts(12),
            in_flight_high_watermark=3,
            in_flight_low_watermark=1,
        )
        thread, outcome = self._start(drive, channels)
        dflash, dspark = (
            StreamingRefChannel(channels[name].path) for name in ("dflash", "dspark")
        )

        # dspark trains on every ref at once; dflash never steps.
        self.assertTrue(
            self._wait(lambda: channels["dflash"].published == 3, train=dspark)
        )
        self._wait(lambda: False, train=dspark, timeout=0.2)
        self.assertEqual(channels["dflash"].published, 3)
        self.assertEqual(channels["dspark"].in_flight_remote(), 0)
        self.assertTrue(thread.is_alive())

        # Once its trainer is done, dflash stops counting and gets no more.
        dflash.mark_consumer_done()
        self.assertTrue(
            self._wait(lambda: channels["dspark"].is_closed(), train=dspark)
        )
        dspark.mark_consumer_done()
        thread.join(10)
        self.assertEqual(outcome, {"produced": 12})
        self.assertEqual(channels["dspark"].published, 12)
        self.assertEqual(channels["dflash"].published, 3)
        self.assertEqual(len(_samples(self.backend, "dflash")), 3)
        (last_output,) = self.server.specs[-1]["outputs"]
        self.assertTrue(last_output["sample_id"].startswith("run-dspark:"))

    def test_a_drafter_can_stop_in_the_middle_of_a_publish(self):
        channels = self._channels()
        trainer = StreamingRefChannel(channels["dflash"].path)

        class _StoppingLane(_DrafterLane):
            """Stops once ``publish_refs`` has read the flag, as when a second
            capture thread marks the drafter stopped in between."""

            @property
            def stopped(self):
                value = self.__dict__.get("flag", False)
                publishing = sys._getframe(1).f_code.co_name == "publish_refs"
                if publishing and not value and self.run_id == "run-dflash":
                    trainer.mark_consumer_done()
                    self.__dict__["flag"] = True
                return value

            @stopped.setter
            def stopped(self, value):
                self.__dict__["flag"] = value

        drive = self._producer(channels, _prompts(4))
        with mock.patch("specforge.launch._DrafterLane", _StoppingLane):
            self.assertEqual(drive(), 4)
        self.assertEqual(channels["dflash"].published, 1)
        self.assertEqual(channels["dspark"].published, 4)

    def test_a_stopped_drafters_unread_samples_still_count_as_resident(self):
        channels = self._channels()
        dflash, dspark = (
            StreamingRefChannel(channels[name].path) for name in ("dflash", "dspark")
        )

        def second_request(number):
            # dspark has trained on the first prompt; dflash read nothing and
            # is done, leaving two samples of 240 bytes behind.
            if number == 2:
                dspark.mark_consumed(len(dspark.poll()))
                dflash.mark_consumer_done()

        self.server.on_request = second_request
        # A dspark sample is 336 bytes. The third prompt would bring the store
        # to 480 + 2 * 336; without dflash's leftovers it would look like 672.
        drive = self._producer(
            channels, _prompts(4), feature_store_max_resident_bytes=1000
        )
        with self.assertRaisesRegex(MemoryError, "projected=1152 > limit=1000"):
            drive()

    def test_a_failed_trainer_fails_every_channel(self):
        channels = self._channels()
        drive = self._producer(
            channels,
            _prompts(6),
            in_flight_high_watermark=2,
            in_flight_low_watermark=1,
        )
        thread, outcome = self._start(drive, channels)
        self.assertTrue(self._wait(lambda: channels["dflash"].published == 2))

        StreamingRefChannel(channels["dspark"].path).mark_consumer_failed("boom")
        thread.join(10)

        self.assertIn("consumer run-dspark failed: boom", str(outcome["error"]))
        for channel in channels.values():
            # A clean end of stream would pass for complete data.
            self.assertFalse(channel.is_closed())
            self.assertIn("boom", channel.failure())

    def test_a_failed_publish_frees_the_other_drafters_refs(self):
        class _BrokenChannel(StreamingRefChannel):
            def publish(self, ref):
                raise OSError("injected publish failure")

        channels = self._channels(dflash_channel=_BrokenChannel)
        drive = self._producer(channels, _prompts(3))

        with self.assertRaisesRegex(OSError, "injected publish failure"):
            drive()

        self.assertEqual(channels["dspark"].published, 0)
        self.assertEqual(self.backend._d, {})
        for channel in channels.values():
            self.assertIn("injected publish failure", channel.failure())


class TestProducerAssemblyForSeveralDrafters(_MultiDrafterCase):
    def setUp(self):
        super().setUp()
        self.root = tempfile.mkdtemp(prefix="multi_drafter_run_")
        recipe = Config.model_validate(
            {
                "run_id": "run",
                "model": {"target_model_path": "target"},
                "data": {"prompts_path": "prompts.jsonl"},
                "drafters": [
                    {"name": name, "draft": PRESET.format(name)} for name in LAYERS
                ],
                "deployment": {
                    "mode": "disaggregated",
                    "disaggregated": {
                        "control_dir": self.root,
                        "backend": "mooncake",
                        "server_urls": [URL],
                    },
                },
            }
        )
        self.cfg = _config_for_role(recipe, "producer")
        runs = [resolve_run(flatten_drafter(recipe, name)) for name in LAYERS]
        self.drafters = [(run.config, run.algorithm) for run in runs]
        self.paths = {
            name: os.path.join(self.root, name, "refs.jsonl") for name in LAYERS
        }

        def adapter(*args, **kwargs):
            kwargs.update(post_fn=self.server, server_info_fn=self.server.server_info)
            return SGLangServerCaptureAdapter(*args, **kwargs)

        patches = {
            "specforge.training.disaggregated._mooncake_store": self.store,
            "specforge.training.assembly._load_input_tools": None,
            "specforge.training.model_loading.resolve_draft_config": None,
        }
        for target, value in patches.items():
            self.enterContext(mock.patch(target, return_value=value))
        self.enterContext(
            mock.patch(
                "specforge.training.disaggregated._producer_capture_metadata",
                side_effect=lambda cfg, algorithm: (
                    list(LAYERS[cfg.run_id.removeprefix("run-")]),
                    HIDDEN,
                    32,
                    32,
                ),
            )
        )
        self.enterContext(
            mock.patch(
                "specforge.inference.adapters.server_capture."
                "SGLangServerCaptureAdapter",
                side_effect=adapter,
            )
        )

    def _build(self, prepare_prompts, build=_build_online):
        return build(
            self.cfg,
            algorithm=self.drafters[0][1],
            build_model_bundle=mock.Mock(),
            prepare_prompts=prepare_prompts,
            optimizer_factory=mock.Mock(),
            logger=mock.Mock(),
            drafters=self.drafters,
        )

    def test_one_producer_feeds_and_cleans_every_drafter(self):
        prepare_prompts = mock.Mock(return_value=_prompts(5))
        run = self._build(prepare_prompts)

        (other,) = prepare_prompts.call_args.kwargs["others"]
        self.assertEqual(other[0].run_id, "run-dspark")
        for path in self.paths.values():
            with open(path + _ONLINE_SCHEDULE_SUFFIX, encoding="utf-8") as stream:
                self.assertEqual(json.load(stream)["num_prompts"], 5)

        def train(path):
            """A trainer that reads every ref and deletes nothing."""
            reader = StreamingRefChannel(path)
            reader.publish_consumer_quantum(1)
            while not reader.is_closed() or reader.poll():
                time.sleep(0.001)
            reader.mark_consumer_done()

        trainers = [
            threading.Thread(target=train, args=(path,), daemon=True)
            for path in self.paths.values()
        ]
        for trainer in trainers:
            trainer.start()
        self.assertEqual(run.run(), 5)

        for path in self.paths.values():
            self.assertEqual(len(StreamingRefChannel(path).poll()), 5)
        self.assertEqual(self.backend._d, {})

    def test_a_producer_that_fails_to_start_tells_every_drafter(self):
        with self.assertRaisesRegex(RuntimeError, "no data"):
            self._build(
                mock.Mock(side_effect=RuntimeError("no data")),
                build=build_disaggregated_run,
            )
        for path in self.paths.values():
            with open(path + ".failed", encoding="utf-8") as stream:
                self.assertIn("RuntimeError: no data", stream.read())

    def test_prompts_pass_every_drafters_filter(self):
        configs = {
            name: Config.model_validate(
                {
                    "draft": PRESET.format(name),
                    "model": {"target_model_path": "target"},
                    "data": {"prompts_path": "/missing/prompts.jsonl"},
                    "deployment": self.cfg.deployment.model_dump(),
                }
            )
            for name in ("eagle3", "dflash")
        }
        eagle3, dflash = (
            (configs[name], REGISTRY.resolve(name), None) for name in configs
        )
        with mock.patch("specforge.data.prompt_builder.prepare_prompt_tasks") as tasks:
            _prepare_prompts(
                eagle3[0],
                None,
                algorithm=eagle3[1],
                draft_config=None,
                others=[dflash],
            )
        self.assertEqual(tasks.call_args.kwargs["min_loss_tokens"], 2)
        self.assertIs(
            tasks.call_args.kwargs["loss_mask_filter"],
            has_consecutive_supervised_tokens,
        )
        self.assertNotEqual(
            _prompt_cache_key(eagle3[0]),
            _prompt_cache_key(eagle3[0], others=[dflash[0]]),
        )


if __name__ == "__main__":
    unittest.main()
