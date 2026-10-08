"""Disaggregated workers must retain defaults and explicit receive overrides."""

import contextlib
import functools
import sys
import tempfile
import types
import unittest
from unittest import mock

from specforge.config import Config
from specforge.runtime.data_plane import mooncake_rdma
from specforge.runtime.data_plane.mooncake_store import _InjectedReplicateConfig
from specforge.training.disaggregated import _mooncake_store
from tests.test_runtime.test_mooncake_rdma import FakeHost
from tests.test_runtime.test_mooncake_store import _FakeMooncakeStore

_ENDPOINTS = {
    "MOONCAKE_METADATA_SERVER": "http://metadata:8080/metadata",
    "MOONCAKE_MASTER_SERVER_ADDR": "master:50051",
}


class DisaggregatedDefaultsTest(unittest.TestCase):
    def test_worker_store_uses_config_unless_environment_overrides_it(self):
        for configured, override, expected in (
            (None, None, "pinned"),
            ("pageable", None, "pageable"),
            ("pageable", "pinned", "pinned"),
            (None, "pageable", "pageable"),
        ):
            with self.subTest(configured=configured, override=override):
                deployment = {
                    "control_dir": "/control",
                    "backend": "mooncake",
                    "server_urls": ["http://capture:30000"],
                }
                if configured is not None:
                    deployment["receive_buffers"] = configured
                cfg = Config.model_validate(
                    {
                        "model": {"target_model_path": "target"},
                        "data": {"prompts_path": "prompts.jsonl"},
                        "deployment": {
                            "mode": "disaggregated",
                            "disaggregated": deployment,
                        },
                    }
                )
                self._assert_worker_mode(cfg, override, expected)

    def test_legacy_environment_worker_uses_pinned_by_default(self):
        cfg = Config.model_validate(
            {
                "model": {"target_model_path": "target"},
                "data": {"hidden_states_path": "/features"},
            }
        )
        self._assert_worker_mode(cfg, None, "pinned")

    def _assert_worker_mode(self, cfg, override, expected):
        env = {
            "MOONCAKE_METADATA_SERVER": "http://metadata:8080/metadata",
            "MOONCAKE_MASTER_SERVER_ADDR": "master:50051",
        }
        if override is not None:
            env["DISAGG_RECEIVE_BUFFERS"] = override
        with (
            mock.patch.dict("os.environ", env, clear=True),
            mock.patch(
                "specforge.runtime.data_plane.mooncake_store._connect_store",
                return_value=(_FakeMooncakeStore(), _InjectedReplicateConfig),
            ),
        ):
            store = _mooncake_store(cfg)
        self.assertEqual(store.receive_buffers, expected)
        self.assertEqual(store._receive_pool is not None, expected != "pageable")


class _RecordingStore(_FakeMooncakeStore):
    setups = []

    def setup(self, **kwargs):
        type(self).setups.append(kwargs)
        return 0


@contextlib.contextmanager
def _fake_mooncake_client():
    # patch.dict(sys.modules) would also drop every module imported meanwhile.
    module = types.ModuleType("mooncake.store")
    module.MooncakeDistributedStore = _RecordingStore
    module.ReplicateConfig = _InjectedReplicateConfig
    fakes = {"mooncake": types.ModuleType("mooncake"), "mooncake.store": module}
    saved = {name: sys.modules.get(name) for name in fakes}
    sys.modules.update(fakes)
    try:
        yield
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class MooncakeStoreTransportTest(unittest.TestCase):
    """The worker store factory never hands Mooncake an empty RDMA device list."""

    def setUp(self):
        _RecordingStore.setups = []
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.host = FakeHost(directory.name)
        self.cfg = Config.model_validate(
            {
                "model": {"target_model_path": "target"},
                "data": {"hidden_states_path": "/features"},
            }
        )

    def _connect(self, environ):
        with (
            _fake_mooncake_client(),
            mock.patch.dict("os.environ", {**_ENDPOINTS, **environ}, clear=True),
            mock.patch(
                "specforge.runtime.data_plane.mooncake_store.resolve_rdma_devices",
                side_effect=functools.partial(
                    mooncake_rdma.resolve_rdma_devices, root=self.host.root
                ),
            ) as resolve,
        ):
            _mooncake_store(self.cfg)
        return _RecordingStore.setups[-1], resolve

    def test_rdma_without_devices_uses_this_nodes_usable_hcas(self):
        self.host.add("mlx5_0")
        self.host.add("mlx5_1", state="1: DOWN")
        setup, _ = self._connect({"MOONCAKE_PROTOCOL": "rdma"})
        self.assertEqual(setup["protocol"], "rdma")
        self.assertEqual(setup["rdma_devices"], "mlx5_0")

    def test_explicit_devices_are_checked_and_tcp_passes_through(self):
        self.host.add("mlx5_0")
        self.host.add("mlx5_7")
        setup, resolve = self._connect(
            {"MOONCAKE_PROTOCOL": "rdma", "MOONCAKE_RDMA_DEVICES": "mlx5_7"}
        )
        self.assertEqual(setup["rdma_devices"], "mlx5_7")
        self.assertEqual(resolve.call_args.args, ("mlx5_7",))
        setup, resolve = self._connect({"MC_FORCE_TCP": "1"})
        self.assertEqual((setup["protocol"], setup["rdma_devices"]), ("tcp", ""))
        resolve.assert_not_called()

        # Mooncake would drop the typo and run on mlx5_0 alone.
        _RecordingStore.setups = []
        with self.assertRaises(RuntimeError) as raised:
            self._connect(
                {"MOONCAKE_PROTOCOL": "rdma", "MOONCAKE_RDMA_DEVICES": "mlx5_0,mlx5_9"}
            )
        self.assertIn(
            "MOONCAKE_RDMA_DEVICES lists RDMA devices that Mooncake cannot use",
            str(raised.exception),
        )
        self.assertIn("mlx5_9: not under", str(raised.exception))
        self.assertEqual(_RecordingStore.setups, [])

    def test_rdma_honours_mooncake_device_filter(self):
        self.host.add("mlx5_0")
        self.host.add("mlx5_1")
        setup, _ = self._connect(
            {"MOONCAKE_PROTOCOL": "rdma", "MC_TE_FILTERS": "mlx5_1"}
        )
        self.assertEqual(setup["rdma_devices"], "mlx5_1")
        with self.assertRaisesRegex(
            RuntimeError, "mlx5_0: excluded by MC_TE_FILTERS='mlx5_1'"
        ):
            self._connect(
                {
                    "MOONCAKE_PROTOCOL": "rdma",
                    "MOONCAKE_RDMA_DEVICES": "mlx5_0,mlx5_1",
                    "MC_TE_FILTERS": "mlx5_1",
                }
            )

    def test_rdma_without_usable_devices_fails_before_setup(self):
        self.host.add("mlx5_0", node=None)
        with self.assertRaises(RuntimeError) as raised:
            self._connect({"MOONCAKE_PROTOCOL": "rdma"})
        message = str(raised.exception)
        self.assertIn(
            "MOONCAKE_PROTOCOL is rdma and MOONCAKE_RDMA_DEVICES is unset, but no "
            "usable RDMA device was found on this host:",
            message,
        )
        self.assertIn("uverbs0 does not exist in this process", message)
        self.assertIn(
            "To use TCP, set MOONCAKE_PROTOCOL=tcp "
            "(deployment.disaggregated.mooncake_protocol or "
            "managed_local.mooncake.protocol) for every capture server and trainer.",
            message,
        )
        self.assertEqual(_RecordingStore.setups, [])

    def test_rdma_rejects_tcp_overrides_before_setup(self):
        self.host.add("mlx5_0")
        for name in ("MC_FORCE_TCP", "MC_USE_TENT"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"this process's environment sets {name}='1'",
                ):
                    self._connect(
                        {
                            "MOONCAKE_PROTOCOL": "rdma",
                            "MOONCAKE_RDMA_DEVICES": "mlx5_0",
                            name: "1",
                        }
                    )
        self.assertEqual(_RecordingStore.setups, [])
