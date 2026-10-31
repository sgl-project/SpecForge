# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""RDMA device resolution for Mooncake transfer clients.

Mooncake does not refuse an unusable RDMA setup by itself. With
``protocol="rdma"`` and an empty device list it auto-discovers HCAs, and the
x86 CUDA wheels install the TCP transport, logging only at INFO, when none is
usable. A listed device that Mooncake cannot use, or that ``MC_TE_FILTERS``
excludes, is dropped without a warning. SpecForge's own clients therefore never
pass Mooncake an empty or unchecked list: they read sysfs, apply the per-device
checks Mooncake applies on the port and GID index it will open, and raise with
the reason for every rejected device. The patched SGLang capture server does
not use this module; it only refuses an empty list.

Torch-free, so the launch supervisor can import it.
"""

from __future__ import annotations

import os
import re
import resource
import stat
from dataclasses import dataclass
from typing import List, Mapping, Optional

#: Tests pass a fake tree holding ``sys/class/infiniband``, ``dev/infiniband``
#: and ``proc/self/status``.
HOST_ROOT = "/"
_SYSFS_DEVICES = os.path.join("sys", "class", "infiniband")
_DEVICE_NODES = os.path.join("dev", "infiniband")
_INFINIBAND = "InfiniBand"
_ETHERNET = "Ethernet"
#: Mooncake opens every HCA on this one port unless MC_IB_PORT names another.
_DEFAULT_IB_PORT = 1
#: Linux capability that exempts memory registration from RLIMIT_MEMLOCK.
_CAP_IPC_LOCK_BIT = 14
_EXPOSE_HCAS = (
    "expose the HCAs to this process (in a container: --device /dev/infiniband "
    "with --ulimit memlock=-1 or --cap-add IPC_LOCK)"
)
#: Mooncake switches that replace the RDMA transport or ignore its device list.
_RDMA_OVERRIDES = (
    ("MC_FORCE_TCP", "installs the TCP transport"),
    (
        "MC_MS_AUTO_DISC",
        "auto-discovers HCAs, ignoring the device list and falling back to TCP "
        "when none is usable",
    ),
    ("MC_USE_TENT", "selects the TENT engine, which ignores the device list"),
    ("MC_USE_TEV1", "selects the TENT engine, which ignores the device list"),
)


@dataclass(frozen=True)
class RdmaDevice:
    """One HCA under ``/sys/class/infiniband``."""

    name: str
    #: ``InfiniBand`` or ``Ethernet`` (RoCE) for a usable device.
    link_layer: Optional[str] = None
    #: Why Mooncake cannot use the device; None when it can.
    problem: Optional[str] = None


def _read(path: str) -> Optional[str]:
    # Unpopulated GID attribute entries fail to read with EINVAL.
    try:
        with open(path, encoding="utf-8") as stream:
            return stream.read().strip()
    except OSError:
        return None


def _listdir(path: str) -> List[str]:
    try:
        return os.listdir(path)
    except OSError:
        return []


def _natural_key(name: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def _leading_int(value: str) -> Optional[int]:
    # Mooncake parses its integer switches with atoi or std::stoi.
    match = re.match(r"\s*[+-]?\d+", value)
    return int(match.group()) if match is not None else None


def _ib_port(env: Mapping[str, str]) -> int:
    """The one port Mooncake opens on every HCA (``MC_IB_PORT``)."""
    value = env.get("MC_IB_PORT")
    if value is None:
        return _DEFAULT_IB_PORT
    port = _leading_int(value) or 0
    # Mooncake ignores an out-of-range value with a warning.
    return port if 0 <= port < 256 else _DEFAULT_IB_PORT


def _gid_index(env: Mapping[str, str]) -> Optional[int]:
    """The GID index Mooncake uses on every port; None selects one per port."""
    value = env.get("MC_GID_INDEX")
    if value is None:
        value = env.get("NCCL_IB_GID_INDEX")
    if value is None:
        return None
    index = _leading_int(value) or 0
    return index if 0 <= index < 256 else None


def _device_whitelist(env: Mapping[str, str]) -> List[str]:
    """Mooncake's ``MC_TE_FILTERS`` whitelist; empty admits every device."""
    return [
        name.strip() for name in env.get("MC_TE_FILTERS", "").split(",") if name.strip()
    ]


def _is_nonzero_gid(gid: Optional[str]) -> bool:
    return bool((gid or "").replace(":", "").strip("0"))


def _has_roce_v2_gid(port_dir: str) -> bool:
    """Mooncake needs a RoCE v2 GID bound to a network interface."""
    types_dir = os.path.join(port_dir, "gid_attrs", "types")
    for index in _listdir(types_dir):
        if _read(os.path.join(types_dir, index)) != "RoCE v2":
            continue
        if not _read(os.path.join(port_dir, "gid_attrs", "ndevs", index)):
            continue
        if _is_nonzero_gid(_read(os.path.join(port_dir, "gids", index))):
            return True
    return False


def _port_problem(
    port_dir: str, port: int, gid_index: Optional[int]
) -> tuple[Optional[str], Optional[str]]:
    """Return ``(link_layer, None)`` for a usable port, else ``(None, problem)``."""
    state = _read(os.path.join(port_dir, "state")) or "unreadable"
    if not state.startswith("4:"):
        return None, f"port {port} is not ACTIVE (state {state!r})"
    link_layer = _read(os.path.join(port_dir, "link_layer"))
    if link_layer not in (_INFINIBAND, _ETHERNET):
        return None, f"port {port} has unsupported link layer {link_layer!r}"
    if gid_index is not None:
        # Mooncake uses exactly this GID and refuses a null one.
        if not _is_nonzero_gid(_read(os.path.join(port_dir, "gids", str(gid_index)))):
            return None, (
                f"port {port} has no GID at index {gid_index} (MC_GID_INDEX or "
                "NCCL_IB_GID_INDEX)"
            )
    elif link_layer == _ETHERNET and not _has_roce_v2_gid(port_dir):
        missing = "without a RoCE v2 GID bound to a network interface"
        return None, f"port {port} is RoCE {missing}"
    return link_layer, None


def _verbs_node_problem(device_dir: str, root: str) -> Optional[str]:
    """Mirror Mooncake's check that the device's uverbs node is usable."""
    verbs_dir = os.path.join(device_dir, "device", "infiniband_verbs")
    nodes = sorted(name for name in _listdir(verbs_dir) if name.startswith("uverbs"))
    if not nodes:
        return f"no uverbs device under {verbs_dir}"
    node = os.path.join(root, _DEVICE_NODES, nodes[0])
    try:
        mode = os.stat(node).st_mode
    except OSError:
        # sysfs shows the host's HCAs even when /dev/infiniband is not mounted.
        return f"{node} does not exist in this process (is /dev/infiniband mounted?)"
    if not stat.S_ISCHR(mode):
        return f"{node} is not a character device"
    if not os.access(node, os.R_OK | os.W_OK):
        return f"{node} is not readable and writable by this process"
    return None


def _probe_device(name: str, root: str, env: Mapping[str, str]) -> RdmaDevice:
    whitelist = _device_whitelist(env)
    if whitelist and name not in whitelist:
        return RdmaDevice(
            name, problem=f"excluded by MC_TE_FILTERS={env['MC_TE_FILTERS']!r}"
        )
    device_dir = os.path.join(root, _SYSFS_DEVICES, name)
    problem = _verbs_node_problem(device_dir, root)
    if problem is not None:
        return RdmaDevice(name, problem=problem)
    ports_dir = os.path.join(device_dir, "ports")
    ports = sorted(_listdir(ports_dir), key=_natural_key)
    port = _ib_port(env)
    if str(port) in ports:
        link_layer, problem = _port_problem(
            os.path.join(ports_dir, str(port)), port, _gid_index(env)
        )
        if problem is None:
            return RdmaDevice(name, link_layer=link_layer)
    else:
        problem = f"has no port {port}"
    others = []
    for other in ports:
        if other != str(port):
            state = _read(os.path.join(ports_dir, other, "state")) or "unreadable"
            others.append(f"port {other} is {state!r}")
    if others:
        # Mooncake disables the whole HCA when this one port is unusable.
        problem += f"; Mooncake opens only port {port} (MC_IB_PORT), and " + (
            ", ".join(others)
        )
    return RdmaDevice(name, problem=problem)


def probe_rdma_devices(
    *, root: str = HOST_ROOT, env: Optional[Mapping[str, str]] = None
) -> List[RdmaDevice]:
    """Every HCA under ``/sys/class/infiniband``, in natural name order.

    *env* supplies the Mooncake switches that decide usability
    (``MC_IB_PORT``, ``MC_GID_INDEX``/``NCCL_IB_GID_INDEX``, ``MC_TE_FILTERS``);
    it defaults to this process's environment.
    """
    env = os.environ if env is None else env
    names = _listdir(os.path.join(root, _SYSFS_DEVICES))
    return [_probe_device(name, root, env) for name in sorted(names, key=_natural_key)]


def resolve_rdma_devices(
    requested: Optional[str],
    *,
    selected_by: str,
    devices_setting: str,
    opt_out: str,
    root: str = HOST_ROOT,
    env: Optional[Mapping[str, str]] = None,
) -> str:
    """Return the comma-separated HCAs a Mooncake RDMA client must use.

    Without *requested*, every usable HCA of one link layer is selected,
    InfiniBand before RoCE, so a client never mixes fabrics. A requested list
    is returned only when every listed device is usable. For the error
    messages, *selected_by* says why RDMA is in use, *devices_setting* names
    the device setting and *opt_out* says how to select TCP instead. *env* is
    the client's environment (see :func:`probe_rdma_devices`).
    """
    devices = probe_rdma_devices(root=root, env=env)
    usable = [device for device in devices if device.problem is None]
    names = list(
        dict.fromkeys(
            name.strip() for name in (requested or "").split(",") if name.strip()
        )
    )
    if not names:
        for link_layer in (_INFINIBAND, _ETHERNET):
            selected = [d.name for d in usable if d.link_layer == link_layer]
            if selected:
                return ",".join(selected)
        reasons = [f"  {device.name}: {device.problem}" for device in devices] or [
            f"  no devices under {os.path.join(root, _SYSFS_DEVICES)}"
        ]
        raise RuntimeError(
            f"{selected_by} and {devices_setting} is unset, but no usable RDMA "
            "device was found on this host:\n"
            + "\n".join(reasons)
            + f"\nTo use TCP, {opt_out}. To use RDMA, {_EXPOSE_HCAS}."
        )

    by_name = {device.name: device for device in devices}
    reasons = []
    for name in names:
        device = by_name.get(name)
        if device is None:
            reasons.append(f"  {name}: not under {os.path.join(root, _SYSFS_DEVICES)}")
        elif device.problem is not None:
            reasons.append(f"  {name}: {device.problem}")
    if reasons:
        available = ", ".join(device.name for device in usable) or "none"
        raise RuntimeError(
            f"{devices_setting} lists RDMA devices that Mooncake cannot use and "
            "would drop without a warning:\n"
            + "\n".join(reasons)
            + f"\nUsable devices on this host: {available}. Fix the list, unset "
            f"it to use every usable device, or, to use TCP, {opt_out}."
        )
    return ",".join(names)


def _forces_auto_discovery(value: Optional[str]) -> bool:
    # Mooncake acts on MC_MS_AUTO_DISC only when it parses as 1.
    return _leading_int(value or "") == 1


def check_rdma_environment(
    env: Mapping[str, str],
    *,
    where: str,
    opt_out: str,
    devices: Optional[str] = None,
) -> None:
    """Reject Mooncake switches that defeat an explicit RDMA device list.

    *devices* is the list the client passes Mooncake; ``MC_TE_FILTERS`` must
    admit every name on it.
    """
    conflicts = [
        f"{name}={env[name]!r} ({effect})"
        for name, effect in _RDMA_OVERRIDES
        if name in env
        and (name != "MC_MS_AUTO_DISC" or _forces_auto_discovery(env[name]))
    ]
    whitelist = _device_whitelist(env)
    listed = [name.strip() for name in (devices or "").split(",") if name.strip()]
    skipped = [name for name in listed if whitelist and name not in whitelist]
    if skipped:
        conflicts.append(
            f"MC_TE_FILTERS={env['MC_TE_FILTERS']!r} (Mooncake skips "
            f"{', '.join(skipped)} of the RDMA device list without a warning)"
        )
    if conflicts:
        raise RuntimeError(
            f"Mooncake RDMA is selected, but {where} sets "
            + "; ".join(conflicts)
            + f". Unset {'it' if len(conflicts) == 1 else 'them'}, or, to use "
            f"TCP, {opt_out}."
        )


def _has_ipc_lock(root: str) -> bool:
    status = _read(os.path.join(root, "proc", "self", "status")) or ""
    for line in status.splitlines():
        if line.startswith("CapEff:"):
            try:
                return bool((int(line.split(":", 1)[1], 16) >> _CAP_IPC_LOCK_BIT) & 1)
            except ValueError:
                return False
    return False


def memlock_shortfall(required_bytes: int, *, root: str = HOST_ROOT) -> Optional[int]:
    """Return the RLIMIT_MEMLOCK soft limit if it cannot cover *required_bytes*.

    None means the registration fits: the soft limit is unlimited or large
    enough, or the process holds CAP_IPC_LOCK, which the kernel checks instead
    of the limit.
    """
    soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft == resource.RLIM_INFINITY or soft >= required_bytes:
        return None
    if _has_ipc_lock(root):
        return None
    return soft


__all__ = [
    "HOST_ROOT",
    "RdmaDevice",
    "check_rdma_environment",
    "memlock_shortfall",
    "probe_rdma_devices",
    "resolve_rdma_devices",
]
