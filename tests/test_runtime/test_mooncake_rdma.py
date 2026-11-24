"""Mooncake RDMA devices are resolved from sysfs and never left to auto-discovery."""

from __future__ import annotations

import os
import resource
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from specforge.runtime.data_plane import mooncake_rdma
from specforge.runtime.data_plane.mooncake_rdma import (
    check_rdma_environment,
    memlock_shortfall,
    probe_rdma_devices,
    resolve_rdma_devices,
)

_ROCE_V2_GID = ("RoCE v2", "eth0", "0000:0000:0000:0000:0000:ffff:0a00:0001")
_ZERO_GID = "0000:0000:0000:0000:0000:0000:0000:0000"


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{value}\n", encoding="utf-8")


class FakeHost:
    """A fake ``sys/class/infiniband`` + ``dev/infiniband`` tree."""

    def __init__(self, root: str) -> None:
        self.root = root
        self._nodes = 0

    def add(
        self,
        name: str,
        *,
        state: str = "4: ACTIVE",
        link_layer: str = "InfiniBand",
        gids=(),
        node: str = "char",
        other_ports=(),
    ) -> Path:
        """Add one HCA with an ``InfiniBand`` port 1 and *other_ports*.

        *other_ports* holds ``(port, state)`` pairs. Return the device's
        ``/dev/infiniband`` node path.
        """
        device = Path(self.root, "sys", "class", "infiniband", name)
        for other, other_state in other_ports:
            _write(device / "ports" / other / "state", other_state)
            _write(device / "ports" / other / "link_layer", "InfiniBand")
        port = device / "ports" / "1"
        _write(port / "state", state)
        _write(port / "link_layer", link_layer)
        for index, (gid_type, ndev, gid) in enumerate(gids):
            _write(port / "gid_attrs" / "types" / str(index), gid_type)
            if ndev is not None:
                _write(port / "gid_attrs" / "ndevs" / str(index), ndev)
            _write(port / "gids" / str(index), gid)
        uverbs = f"uverbs{self._nodes}"
        self._nodes += 1
        (device / "device" / "infiniband_verbs" / uverbs).mkdir(parents=True)
        path = Path(self.root, "dev", "infiniband", uverbs)
        path.parent.mkdir(parents=True, exist_ok=True)
        if node == "char":
            # /dev/null is a readable and writable character device.
            os.symlink(os.devnull, path)
        elif node == "file":
            path.write_text("", encoding="utf-8")
        return path

    def resolve(self, requested=None, env=None) -> str:
        return resolve_rdma_devices(
            requested,
            selected_by="protocol is rdma",
            devices_setting="rdma_devices",
            opt_out="set protocol: tcp",
            root=self.root,
            # Keep this host's Mooncake switches out of the fake one.
            env=env or {},
        )


class MooncakeRdmaDeviceTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.host = FakeHost(directory.name)

    def test_active_infiniband_devices_are_selected_in_natural_order(self):
        for name in ("mlx5_10", "mlx5_2", "mlx5_0"):
            self.host.add(name)
        self.assertEqual(self.host.resolve(), "mlx5_0,mlx5_2,mlx5_10")

    def test_down_port_is_not_usable(self):
        self.host.add("mlx5_0", state="1: DOWN")
        with self.assertRaises(RuntimeError) as raised:
            self.host.resolve()
        message = str(raised.exception)
        self.assertIn(
            "protocol is rdma and rdma_devices is unset, but no usable RDMA device "
            "was found on this host:",
            message,
        )
        self.assertIn("mlx5_0: port 1 is not ACTIVE (state '1: DOWN')", message)
        self.assertIn("To use TCP, set protocol: tcp.", message)
        self.assertIn("--device /dev/infiniband", message)
        self.assertIn("--ulimit memlock=-1", message)

    def test_roce_needs_a_roce_v2_gid_on_a_network_interface(self):
        without = {
            "v1 only": [("IB/RoCE v1", "eth0", _ROCE_V2_GID[2])],
            "no netdev": [("RoCE v2", None, _ROCE_V2_GID[2])],
            "zero gid": [("RoCE v2", "eth0", _ZERO_GID)],
        }
        for case, gids in without.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as root:
                host = FakeHost(root)
                host.add("mlx5_0", link_layer="Ethernet", gids=gids)
                with self.assertRaisesRegex(
                    RuntimeError, "port 1 is RoCE without a RoCE v2 GID"
                ):
                    host.resolve()
        self.host.add(
            "mlx5_0",
            link_layer="Ethernet",
            gids=[("IB/RoCE v1", "eth0", _ROCE_V2_GID[2]), _ROCE_V2_GID],
        )
        self.assertEqual(self.host.resolve(), "mlx5_0")

    def test_missing_device_node_is_not_usable(self):
        # Containers see the host's HCAs in sysfs without /dev/infiniband.
        self.host.add("mlx5_0", node=None)
        with self.assertRaisesRegex(
            RuntimeError,
            r"mlx5_0: .*uverbs0 does not exist in this process "
            r"\(is /dev/infiniband mounted\?\)",
        ):
            self.host.resolve()

    def test_device_node_must_be_a_readable_writable_character_device(self):
        self.host.add("mlx5_0", node="file")
        with self.assertRaisesRegex(RuntimeError, "is not a character device"):
            self.host.resolve()

        with tempfile.TemporaryDirectory() as root:
            host = FakeHost(root)
            node = str(host.add("mlx5_0"))
            access = os.access
            with mock.patch.object(
                mooncake_rdma.os,
                "access",
                side_effect=lambda path, mode: path != node and access(path, mode),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "is not readable and writable by this process"
                ):
                    host.resolve()

    def test_infiniband_wins_and_link_layers_never_mix(self):
        self.host.add("mlx5_0", link_layer="Ethernet", gids=[_ROCE_V2_GID])
        self.host.add("mlx5_1")
        self.host.add("mlx5_2")
        self.assertEqual(
            [
                (device.name, device.link_layer)
                for device in probe_rdma_devices(root=self.host.root, env={})
            ],
            [
                ("mlx5_0", "Ethernet"),
                ("mlx5_1", "InfiniBand"),
                ("mlx5_2", "InfiniBand"),
            ],
        )
        self.assertEqual(self.host.resolve(), "mlx5_1,mlx5_2")

    def test_listed_devices_are_validated_by_name(self):
        self.host.add("mlx5_0")
        self.host.add("mlx5_1", state="1: DOWN")
        self.assertEqual(self.host.resolve(" mlx5_0 ,mlx5_0"), "mlx5_0")
        with self.assertRaises(RuntimeError) as raised:
            self.host.resolve("mlx5_0,mlx5_9,mlx5_1")
        message = str(raised.exception)
        self.assertIn(
            "rdma_devices lists RDMA devices that Mooncake cannot use and would "
            "drop without a warning:",
            message,
        )
        self.assertIn("mlx5_9: not under", message)
        self.assertIn("mlx5_1: port 1 is not ACTIVE", message)
        self.assertNotIn("mlx5_0:", message)
        self.assertIn("Usable devices on this host: mlx5_0.", message)

    def test_host_without_rdma_devices_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, "no devices under .*infiniband"):
            self.host.resolve()

    def test_only_the_port_mooncake_opens_counts(self):
        # Mooncake opens every HCA on MC_IB_PORT (default 1) and disables the
        # whole HCA when that port is down, whatever its other ports do.
        self.host.add("mlx4_0", state="1: DOWN", other_ports=[("2", "4: ACTIVE")])
        with self.assertRaises(RuntimeError) as raised:
            self.host.resolve()
        self.assertIn(
            "mlx4_0: port 1 is not ACTIVE (state '1: DOWN'); Mooncake opens only "
            "port 1 (MC_IB_PORT), and port 2 is '4: ACTIVE'",
            str(raised.exception),
        )
        self.assertEqual(self.host.resolve(env={"MC_IB_PORT": "2"}), "mlx4_0")
        with self.assertRaisesRegex(RuntimeError, "mlx4_0: has no port 3;"):
            self.host.resolve(env={"MC_IB_PORT": "3"})
        # Mooncake ignores an out-of-range port and keeps port 1.
        with self.assertRaisesRegex(RuntimeError, "port 1 is not ACTIVE"):
            self.host.resolve(env={"MC_IB_PORT": "300"})

    def test_a_configured_gid_index_must_hold_a_gid(self):
        gids = [("IB/RoCE v1", None, _ROCE_V2_GID[2]), ("RoCE v2", "eth0", _ZERO_GID)]
        self.host.add("mlx5_0", link_layer="Ethernet", gids=gids)
        # No usable RoCE v2 GID for Mooncake's own selection ...
        with self.assertRaisesRegex(RuntimeError, "without a RoCE v2 GID"):
            self.host.resolve()
        # ... but a configured index is used as is, and only a null GID fails.
        self.assertEqual(self.host.resolve(env={"MC_GID_INDEX": "0"}), "mlx5_0")
        for env in (
            {"MC_GID_INDEX": "1"},
            {"NCCL_IB_GID_INDEX": "1"},
            {"MC_GID_INDEX": "7", "NCCL_IB_GID_INDEX": "0"},
        ):
            with self.subTest(env=env):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"port 1 has no GID at index \d \(MC_GID_INDEX or "
                    r"NCCL_IB_GID_INDEX\)",
                ):
                    self.host.resolve(env=env)

    def test_mooncake_device_filter_limits_selection_and_listed_devices(self):
        for name in ("mlx5_0", "mlx5_1", "mlx5_2"):
            self.host.add(name)
        env = {"MC_TE_FILTERS": " mlx5_2, mlx5_0 "}
        self.assertEqual(self.host.resolve(env=env), "mlx5_0,mlx5_2")
        with self.assertRaises(RuntimeError) as raised:
            self.host.resolve("mlx5_0,mlx5_1", env=env)
        self.assertIn(
            "mlx5_1: excluded by MC_TE_FILTERS=' mlx5_2, mlx5_0 '",
            str(raised.exception),
        )
        with self.assertRaisesRegex(RuntimeError, "no usable RDMA device"):
            self.host.resolve(env={"MC_TE_FILTERS": "mlx5_9"})
        # An empty whitelist admits every device, as in Mooncake.
        self.assertEqual(
            self.host.resolve(env={"MC_TE_FILTERS": " , "}), "mlx5_0,mlx5_1,mlx5_2"
        )


class MooncakeRdmaEnvironmentTest(unittest.TestCase):
    def test_overrides_that_force_tcp_or_ignore_devices_are_rejected(self):
        for name, value in (
            ("MC_FORCE_TCP", ""),
            ("MC_MS_AUTO_DISC", "1"),
            ("MC_USE_TENT", "1"),
            ("MC_USE_TEV1", "0"),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"Mooncake RDMA is selected, but the trainer environment sets "
                    rf"{name}='{value}' .*Unset it, or, to use TCP, set protocol: tcp",
                ):
                    check_rdma_environment(
                        {name: value},
                        where="the trainer environment",
                        opt_out="set protocol: tcp",
                    )

    def test_device_filter_must_admit_every_listed_device(self):
        environment = {"MC_TE_FILTERS": "mlx5_0,mlx5_2"}
        check_rdma_environment(
            environment, where="env", opt_out="set tcp", devices="mlx5_0,mlx5_2"
        )
        check_rdma_environment(environment, where="env", opt_out="set tcp")
        with self.assertRaisesRegex(
            RuntimeError,
            r"env sets MC_TE_FILTERS='mlx5_0,mlx5_2' \(Mooncake skips mlx5_1, "
            r"mlx5_3 of the RDMA device list without a warning\)\. Unset it",
        ):
            check_rdma_environment(
                environment,
                where="env",
                opt_out="set tcp",
                devices="mlx5_0,mlx5_1,mlx5_3",
            )

    def test_auto_discovery_is_rejected_only_when_enabled(self):
        for value in ("0", "", "yes"):
            with self.subTest(value=value):
                check_rdma_environment(
                    {"MC_MS_AUTO_DISC": value}, where="env", opt_out="set tcp"
                )
        with self.assertRaisesRegex(RuntimeError, "Unset them"):
            check_rdma_environment(
                {"MC_MS_AUTO_DISC": " 1", "MC_FORCE_TCP": "1"},
                where="env",
                opt_out="set tcp",
            )

    def test_memlock_limit_must_cover_registration_unless_exempt(self):
        with tempfile.TemporaryDirectory() as root:
            status = Path(root, "proc", "self", "status")
            _write(status, "Name:\tpython\nCapEff:\t0000000000000000")
            for soft, required, expected in (
                (resource.RLIM_INFINITY, 1 << 40, None),
                (1 << 30, 1 << 30, None),
                (64 << 10, 1 << 30, 64 << 10),
            ):
                with (
                    self.subTest(soft=soft),
                    mock.patch.object(
                        mooncake_rdma.resource,
                        "getrlimit",
                        return_value=(soft, soft),
                    ),
                ):
                    self.assertEqual(memlock_shortfall(required, root=root), expected)
            # CAP_IPC_LOCK (bit 14) exempts registration from the limit.
            _write(status, "CapEff:\t0000000000004000")
            with mock.patch.object(
                mooncake_rdma.resource, "getrlimit", return_value=(64 << 10, 64 << 10)
            ):
                self.assertIsNone(memlock_shortfall(1 << 30, root=root))


if __name__ == "__main__":
    unittest.main()
