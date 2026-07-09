# coding=utf-8
"""Multi-server producer fan-out (no GPU, no server, no mooncake master).

The multi-server topology: ``build_disagg_online_producer(feature_source=[...])``
builds one RolloutWorker per SGLangServerCaptureAdapter (1 server : 1 adapter :
1 worker), all leasing DISJOINT prompts from the one controller and publishing
into the one channel, concurrently. Each stub ``post_fn`` stands in for one
patched SGLang server writing into the SHARED fake Mooncake backend — exactly
the shape of ``examples/disagg/run_qwen3.6_27b_dflash_disagg_multiserver.sh``.

Covers the failure matrix the single-server path never hits:
- disjoint + complete production across two live servers;
- one dead server: its leases fail retryable, the survivor re-leases and
  finishes the pool (no truncation, no hang);
- all servers dead: loud RuntimeError, channel still closed;
- a poisoned prompt (server rejects it every time): terminal after
  ``max_prompt_attempts`` instead of spinning the drained-detection loop.
"""

import os
import tempfile
import threading
import time
import unittest

from specforge.inference.adapters.server_capture import SGLangServerCaptureAdapter
from specforge.launch import build_disagg_online_producer
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel
from tests.test_runtime.test_server_capture import (
    AUX_LAYERS,
    HIDDEN,
    _FakeMooncakeStore,
    _StubCaptureServer,
)


def _SLEEP(_s):  # injected drive sleep: keep retry/backpressure spins bounded
    time.sleep(0.001)


def _prompts(n, seq=6):
    return [
        {
            "task_id": f"p{i}",
            "payload": {
                "input_ids": list(range(1, seq + 1 + i)),
                "loss_mask": [1] * (seq + i),
            },
        }
        for i in range(n)
    ]


def _adapter(store, post_fn, url="http://server:30000"):
    return SGLangServerCaptureAdapter(
        url, store, run_id="run0", strategy="dflash", post_fn=post_fn
    )


def _build(adapters, prompts, store, channel, **kw):
    return build_disagg_online_producer(
        strategy="dflash",
        feature_source=adapters,
        prompts=prompts,
        feature_store=store,
        channel=channel,
        run_id="run0",
        target_hidden_size=HIDDEN,
        target_repr=None,
        aux_hidden_state_layer_ids=AUX_LAYERS,
        sleep=_SLEEP,
        **kw,
    )


def _published_refs(path):
    return StreamingRefChannel(path).poll()


def _published_sample_ids(path):
    return [r.sample_id for r in _published_refs(path)]


class TestMultiServerProducer(unittest.TestCase):
    def _workdir(self):
        return tempfile.mkdtemp(prefix="disagg_multisrv_")

    def test_two_servers_disjoint_and_complete(self):
        backend = _FakeMooncakeStore()
        stubs = [_StubCaptureServer(backend), _StubCaptureServer(backend)]
        store = MooncakeFeatureStore(store=backend, store_id="run0")
        adapters = [
            _adapter(store, stubs[i], url=f"http://server{i}:3000{i}") for i in range(2)
        ]
        N = 12
        channel = StreamingRefChannel(os.path.join(self._workdir(), "refs.jsonl"))
        workers, drive = _build(adapters, _prompts(N), store, channel, lease=2)
        self.assertEqual(len(workers), 2)

        produced = drive()
        self.assertEqual(produced, N)
        self.assertTrue(channel.is_closed())

        ids = _published_sample_ids(channel.path)
        self.assertEqual(len(ids), N)
        self.assertEqual(len(set(ids)), N)  # no duplicate publishes

        seen0, seen1 = set(stubs[0].expected), set(stubs[1].expected)
        self.assertEqual(seen0 | seen1, {f"run0:p{i}" for i in range(N)})
        self.assertEqual(seen0 & seen1, set())  # disjoint prompt slices
        # concurrency is real: both servers captured (lease=2 over 12 prompts
        # leaves plenty for the peer even in the worst interleaving)
        self.assertTrue(seen0 and seen1)

        # ref-level provenance mirrors the stub-level split (the post-hoc
        # audit trail a real multi-server run relies on)
        by_server = {}
        for r in _published_refs(channel.path):
            by_server.setdefault(r.metadata["server"], set()).add(r.sample_id)
        self.assertEqual(by_server.get("http://server0:30000"), seen0)
        self.assertEqual(by_server.get("http://server1:30001"), seen1)

    def test_prompt_epochs_republish_with_unique_sample_ids(self):
        backend = _FakeMooncakeStore()
        stub = _StubCaptureServer(backend)
        store = MooncakeFeatureStore(store=backend, store_id="run0")
        N, E = 3, 2
        channel = StreamingRefChannel(os.path.join(self._workdir(), "refs.jsonl"))
        _workers, drive = _build(
            [_adapter(store, stub)],
            _prompts(N),
            store,
            channel,
            lease=2,
            prompt_epochs=E,
        )

        produced = drive()
        self.assertEqual(produced, N * E)
        ids = _published_sample_ids(channel.path)
        self.assertEqual(len(ids), N * E)
        self.assertEqual(len(set(ids)), N * E)
        self.assertEqual(
            set(ids),
            {
                f"run0:epoch{epoch:04d}-prompt{idx:012d}"
                for epoch in range(E)
                for idx in range(N)
            },
        )
        self.assertTrue(channel.is_closed())

    def test_one_dead_server_survivor_completes_pool(self):
        backend = _FakeMooncakeStore()
        healthy = _StubCaptureServer(backend)
        store = MooncakeFeatureStore(store=backend, store_id="run0")

        dead_leased = threading.Event()

        def dead_post(url, json_body, timeout):
            dead_leased.set()  # it held leases when it died
            raise ConnectionError("server 1 unreachable")

        def healthy_post(url, json_body, timeout):
            dead_leased.wait(5)  # let the dead server lease + fail first
            return healthy(url, json_body, timeout)

        adapters = [
            _adapter(store, healthy_post, url="http://server0:30000"),
            _adapter(store, dead_post, url="http://server1:30001"),
        ]
        N = 8
        channel = StreamingRefChannel(os.path.join(self._workdir(), "refs.jsonl"))
        workers, drive = _build(adapters, _prompts(N), store, channel, lease=2)

        produced = drive(max_rounds=10_000)
        # the dead worker's leases were failed retryable and re-leased by the
        # survivor: every prompt still becomes exactly one ref.
        self.assertEqual(produced, N)
        self.assertTrue(dead_leased.is_set())
        self.assertEqual(set(healthy.expected), {f"run0:p{i}" for i in range(N)})
        ids = _published_sample_ids(channel.path)
        self.assertEqual(sorted(ids), sorted(set(ids)))
        self.assertTrue(channel.is_closed())
        self.assertTrue(workers[1].health()["recent_failures"])

    def test_all_servers_dead_raises_loudly(self):
        backend = _FakeMooncakeStore()
        store = MooncakeFeatureStore(store=backend, store_id="run0")

        def dead_post(url, json_body, timeout):
            raise ConnectionError("pool down")

        adapters = [
            _adapter(store, dead_post, url=f"http://server{i}:3000{i}")
            for i in range(2)
        ]
        channel = StreamingRefChannel(os.path.join(self._workdir(), "refs.jsonl"))
        _workers, drive = _build(
            adapters, _prompts(4), store, channel, lease=2, max_worker_failures=2
        )
        with self.assertRaises(RuntimeError):
            drive(max_rounds=10_000)
        # EOF still reaches the consumer — a crashed producer must not hang it.
        self.assertTrue(channel.is_closed())

    def test_poisoned_prompt_goes_terminal_not_infinite(self):
        # PR654 known rough edge: an all-failed round used to read as
        # pool-drained (silent truncation); with drained = pending==0 AND
        # leased==0 the loop instead retries — so a prompt the server rejects
        # every time must go terminal via max_prompt_attempts, not spin.
        backend = _FakeMooncakeStore()
        stub = _StubCaptureServer(backend, error_sample_ids={"run0:p0"})
        store = MooncakeFeatureStore(store=backend, store_id="run0")
        N = 5
        channel = StreamingRefChannel(os.path.join(self._workdir(), "refs.jsonl"))
        _workers, drive = _build(
            [_adapter(store, stub)],
            _prompts(N),
            store,
            channel,
            lease=2,
            max_prompt_attempts=3,
        )
        produced = drive(max_rounds=10_000)
        self.assertEqual(produced, N - 1)  # p0 terminal, everything else through
        ids = _published_sample_ids(channel.path)
        self.assertEqual(set(ids), {f"run0:p{i}" for i in range(1, N)})
        self.assertTrue(channel.is_closed())

    def test_source_count_worker_count_conflict_raises(self):
        backend = _FakeMooncakeStore()
        store = MooncakeFeatureStore(store=backend, store_id="run0")
        stub = _StubCaptureServer(backend)
        adapters = [_adapter(store, stub), _adapter(store, stub)]
        channel = StreamingRefChannel(os.path.join(self._workdir(), "refs.jsonl"))
        with self.assertRaises(ValueError):
            _build(adapters, _prompts(2), store, channel, num_rollout_workers=3)


if __name__ == "__main__":
    unittest.main()
