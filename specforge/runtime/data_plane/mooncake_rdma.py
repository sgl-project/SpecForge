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
usable. A listed device that Mooncake cannot use is dropped without a warning.
SpecForge therefore never passes Mooncake an empty or unchecked list: it reads
sysfs itself, applies the per-device checks of Mooncake's topology discovery,
and raises with the reason for every rejected device.

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


def _has_roce_v2_gid(port_dir: str) -> bool:
    """Mooncake needs a RoCE v2 GID bound to a network interface."""
    types_dir = os.path.join(port_dir, "gid_attrs", "types")
    for index in _listdir(types_dir):
        if _read(os.path.join(types_dir, index)) != "RoCE v2":
            continue
        if not _read(os.path.join(port_dir, "gid_attrs", "ndevs", index)):
            continue
        gid = _read(os.path.join(port_dir, "gids", index)) or ""
        if gid.replace(":", "").strip("0"):
            return True
    return False


def _port_problem(port_dir: str, port: str) -> tuple[Optional[str], Optional[str]]:
    """Return ``(link_layer, None)`` for a usable port, else ``(None, problem)``."""
    state = _read(os.path.join(port_dir, "state")) or "unreadable"
    if not state.startswith("4:"):
        return None, f"port {port} is not ACTIVE (state {state!r})"
    link_layer = _read(os.path.join(port_dir, "link_layer"))
    if link_layer == _INFINIBAND:
        return link_layer, None
    if link_layer == _ETHERNET:
        if _has_roce_v2_gid(port_dir):
            return link_layer, None
        missing = "without a RoCE v2 GID bound to a network interface"
        return None, f"port {port} is RoCE {missing}"
    return None, f"port {port} has unsupported link layer {link_layer!r}"


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


def _probe_device(name: str, root: str) -> RdmaDevice:
    device_dir = os.path.join(root, _SYSFS_DEVICES, name)
    problem = _verbs_node_problem(device_dir, root)
    if problem is not None:
        return RdmaDevice(name, problem=problem)
    ports_dir = os.path.join(device_dir, "ports")
    link_layers = []
    problems = []
    for port in sorted(_listdir(ports_dir), key=_natural_key):
        link_layer, problem = _port_problem(os.path.join(ports_dir, port), port)
        if link_layer is not None:
            link_layers.append(link_layer)
        else:
            problems.append(problem)
    if not link_layers:
        return RdmaDevice(name, problem="; ".join(problems) or "has no ports")
    return RdmaDevice(
        name, link_layer=_INFINIBAND if _INFINIBAND in link_layers else _ETHERNET
    )


def probe_rdma_devices(*, root: str = HOST_ROOT) -> List[RdmaDevice]:
    """Every HCA under ``/sys/class/infiniband``, in natural name order."""
    names = _listdir(os.path.join(root, _SYSFS_DEVICES))
    return [_probe_device(name, root) for name in sorted(names, key=_natural_key)]


def resolve_rdma_devices(
    requested: Optional[str],
    *,
    selected_by: str,
    devices_setting: str,
    opt_out: str,
    root: str = HOST_ROOT,
) -> str:
    """Return the comma-separated HCAs a Mooncake RDMA client must use.

    Without *requested*, every usable HCA of one link layer is selected,
    InfiniBand before RoCE, so a client never mixes fabrics. A requested list
    is returned only when every listed device is usable. For the error
    messages, *selected_by* says why RDMA is in use, *devices_setting* names
    the device setting and *opt_out* says how to select TCP instead.
    """
    devices = probe_rdma_devices(root=root)
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
    # Mooncake parses MC_MS_AUTO_DISC with std::stoi and acts only on 1.
    match = re.match(r"\s*[+-]?\d+", value or "")
    return match is not None and int(match.group()) == 1


def check_rdma_environment(env: Mapping[str, str], *, where: str, opt_out: str) -> None:
    """Reject Mooncake switches that defeat an explicit RDMA device list."""
    conflicts = [
        f"{name}={env[name]!r} ({effect})"
        for name, effect in _RDMA_OVERRIDES
        if name in env
        and (name != "MC_MS_AUTO_DISC" or _forces_auto_discovery(env[name]))
    ]
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
