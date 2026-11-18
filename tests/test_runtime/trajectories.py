"""Record what ``specforge train`` does, update by update, on tiny synthetic runs.

A recording holds, for every optimizer update of every case, what each rank
logged, the learning rate its optimizer applied, and the checkpoint written
after it: draft weights, every rank's optimizer and RNG state, and the data
position. Where the checkpoint stores one replica for all ranks (DDP), it also
holds each rank's own weights and optimizer state after the update. Comparing
two recordings names the values that differ, which is how a change to the
training step is shown to leave training unchanged:

    python -m tests.test_runtime.trajectories record before.pt   # on the base
    python -m tests.test_runtime.trajectories record after.pt    # on the change
    python -m tests.test_runtime.trajectories compare before.pt after.pt

Both recordings have to come from the same machine and torch build: across
those, floating-point results differ in the last bits.
"""

import argparse
import os
import re
import socket
import struct
import sys
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.multiprocessing as multiprocessing
from torch.nn.modules.module import register_module_forward_pre_hook
from torch.nn.parallel import DistributedDataParallel
from torch.optim.optimizer import (
    register_optimizer_step_post_hook,
    register_optimizer_step_pre_hook,
)

from specforge.training.checkpoint import STATE_FILE
from specforge.utils import get_device_type
from tests.test_runtime.test_canonical_end_to_end import drafts, write_run

EPOCHS = 2
BATCHES_PER_EPOCH = 3
# World size, accumulation window, wrapper, parameter dtype. Under bfloat16
# the optimizer's fp32 masters hold bits the parameters do not.
TOPOLOGIES = (
    (1, 1, "NO_SHARD", "float32"),
    (1, 2, "NO_SHARD", "float32"),
    (2, 2, "NO_SHARD", "float32"),
    (2, 2, "NO_SHARD", "bfloat16"),
    (2, 2, "SHARD_GRAD_OP", "float32"),
    (4, 2, "SHARD_GRAD_OP", "bfloat16"),
)
# What a checkpoint's shared file says about the run's progress and weights.
SHARED_STATE = (
    "draft_state_dict",
    "replicated_optimizer_state",
    "global_step",
    "epoch",
    "epoch_batch",
    "epoch_samples",
)
CHECKPOINT = re.compile(r"specforge-run-step(\d+)")


@dataclass(frozen=True)
class Case:
    """One run: a draft architecture under one topology and accumulation window."""

    architecture: str
    world_size: int = 1
    accumulation_steps: int = 1
    sharding: str = "NO_SHARD"
    dtype: str = "float32"

    @property
    def name(self):
        return (
            f"{self.architecture}/ranks{self.world_size}"
            f"/accumulation{self.accumulation_steps}/{self.sharding}/{self.dtype}"
        )

    @property
    def updates(self):
        """Optimizer updates the run reaches: windows span the epoch boundary."""
        return EPOCHS * BATCHES_PER_EPOCH // self.accumulation_steps


def every_case(ranks=()):
    """Every architecture under every topology, and under each extra world size."""
    topologies = TOPOLOGIES + tuple(
        topology
        for world_size in ranks
        for topology in (
            (world_size, 2, "NO_SHARD", "float32"),
            (world_size, 2, "SHARD_GRAD_OP", "bfloat16"),
        )
    )
    return [
        Case(draft.architecture, *topology)
        for draft in drafts()
        for topology in dict.fromkeys(topologies)
    ]


def cases():
    """The cases this host can place.

    An accelerator run needs one device per rank. FSDP refuses a CPU model
    unless the host has an accelerator to compute on, so a CPU trainer records
    the DDP cases only.
    """
    device_type = get_device_type()
    if device_type == "cpu":
        return [case for case in every_case() if case.sharding == "NO_SHARD"]
    devices = torch.get_device_module(device_type).device_count()
    return [case for case in every_case() if case.world_size <= devices]


def _vary_supervision(features):
    """Give the records different supervised lengths.

    Ranks and micro-batches then hold different loss denominators, so a
    denominator pooled the wrong way changes the update.
    """
    for index, path in enumerate(sorted(features.iterdir())):
        record = torch.load(path)
        supervised = record["loss_mask"].nonzero().flatten()
        record["loss_mask"][supervised[len(supervised) - index % 3 :]] = 0
        torch.save(record, path)


def _free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _rendezvous_port(output, rank):
    """Rank 0 picks the port just before its run binds it; the others read it."""
    path = Path(output) / "rendezvous-port"
    if rank == 0:
        port = _free_port()
        pending = path.with_suffix(".pending")
        pending.write_text(str(port))
        pending.replace(path)
        return port
    deadline = time.monotonic() + 120
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"rank 0 published no rendezvous port in {output}")
        time.sleep(0.01)
    return int(path.read_text())


def _on_host(value):
    """A detached host copy of nested optimizer state."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _on_host(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_on_host(item) for item in value)
    return value


class _Replica:
    """What this rank's own process does, seen through torch's global hooks.

    Every update: the learning rate its optimizer step applied. Under DDP,
    which writes one rank's weights and optimizer for all of them, also this
    rank's optimizer state and weights after the update: a replica that
    drifted would otherwise show only in a later update's pooled loss, and
    not at all where the drift never reaches a later forward.
    """

    def __init__(self):
        self.applied_lr = []
        self.updates = []
        self._model = None
        self._weights_due = False

    def __enter__(self):
        self._hooks = [
            register_module_forward_pre_hook(self._before_forward),
            register_optimizer_step_pre_hook(self._before_step),
            register_optimizer_step_post_hook(self._after_step),
        ]
        return self

    def __exit__(self, *exception):
        for hook in self._hooks:
            hook.remove()

    def _before_forward(self, module, arguments):
        if isinstance(module, DistributedDataParallel):
            self._model = module.module
            self._keep_weights()

    def _before_step(self, optimizer, arguments, keywords):
        self.applied_lr.append([group["lr"] for group in optimizer.param_groups])

    def _after_step(self, optimizer, arguments, keywords):
        if self._model is None:
            return
        self.updates.append(
            {
                "optimizer": _on_host(optimizer.state_dict()),
                "optimized": [
                    _on_host(parameter)
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                ],
            }
        )
        # The trainer copies the optimized tensors into the model after the
        # step; the weights are read once that is done.
        self._weights_due = True

    def _keep_weights(self):
        if self._weights_due:
            self.updates[-1]["weights"] = _on_host(self._model.state_dict())
            self._weights_due = False

    def ended(self):
        """Everything seen, once training has returned."""
        self._keep_weights()
        return {"applied_lr": self.applied_lr, "updates": self.updates}


def _train_rank(rank, world_size, runs):
    """Run every case on one rank, as a worker ``torchrun`` started would."""
    from specforge.cli import main

    # Ranks share this host's cores; one thread each keeps the sums in one order.
    torch.set_num_threads(1)
    for config, overrides, output in runs:
        logged = []
        rendezvous = {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(_rendezvous_port(output, rank)),
            # Set by a torchrun this recorder may itself run under; it would
            # make rank 0 look for a store instead of hosting one.
            "TORCHELASTIC_USE_AGENT_STORE": "False",
        }
        with (
            patch.dict(os.environ, rendezvous),
            # The console logger every rank keeps when no tracker is configured.
            patch(
                "specforge.training.assembly._logger",
                lambda metrics, step: logged.append((step, dict(metrics))),
            ),
            _Replica() as replica,
        ):
            status = main(["train", "-c", config, *overrides])
            if status != 0:
                raise RuntimeError(f"specforge train exited with {status}: {config}")
            ended = {"logged": logged, "replica": replica.ended()}
        torch.save(ended, Path(output) / f"ended-rank{rank}.pt")


def _logged(events, case):
    """Per update, what the rank logged: exactly one event for each update."""
    steps = [step for step, _ in events]
    if steps != list(range(1, len(steps) + 1)):
        raise RuntimeError(f"{case.name}: a rank logged updates {steps}")
    return [
        {
            # Computed on the host; the same on every device.
            "lr": metrics["lr"],
            # Computed on the training device, like the weights.
            "metrics": {
                name: torch.tensor(value, dtype=torch.float64)
                for name, value in metrics.items()
                if name != "lr" and not name.startswith("perf/")
            },
        }
        for _, metrics in events
    ]


def _read_checkpoint(directory, world_size):
    """Read one checkpoint as written: the shared file and every rank's file."""
    shared = torch.load(directory / STATE_FILE, map_location="cpu", weights_only=False)
    return {
        "shared": {name: shared[name] for name in SHARED_STATE if name in shared},
        "ranks": [
            torch.load(
                directory / f"training_state_rank{rank}.pt",
                map_location="cpu",
                weights_only=False,
            )
            for rank in range(world_size)
        ],
    }


def _read_case(output, case):
    """Everything one run left: every checkpoint on disk, every rank's record."""
    ended = [
        torch.load(
            output / f"ended-rank{rank}.pt", map_location="cpu", weights_only=False
        )
        for rank in range(case.world_size)
    ]
    logged = [_logged(rank["logged"], case) for rank in ended]
    steps = sorted(
        int(match.group(1))
        for match in map(CHECKPOINT.fullmatch, os.listdir(output))
        if match
    )
    for rank in ended:
        replica = rank["replica"]
        if len(rank["logged"]) != len(steps) or len(replica["applied_lr"]) != len(
            steps
        ):
            raise RuntimeError(
                f"{case.name}: checkpoints {steps} do not match the "
                f"{len(rank['logged'])} logged and {len(replica['applied_lr'])} "
                "applied updates of a rank"
            )
    return [
        {
            "checkpoint": step,
            "logged": [rank[index] for rank in logged],
            "applied_lr": [rank["replica"]["applied_lr"][index] for rank in ended],
            # Empty unless DDP wrapped the model.
            "replicas": [
                rank["replica"]["updates"][index]
                for rank in ended
                if rank["replica"]["updates"]
            ],
            **_read_checkpoint(output / f"specforge-run-step{step}", case.world_size),
        }
        for index, step in enumerate(steps)
    ]


def record(selected):
    """Train every case and return ``{case name: what the run left}``.

    Cases of one world size share their worker processes, so a recording pays
    for the Python imports once per rank.
    """
    by_name = {draft.architecture: draft for draft in drafts()}
    recording = {}
    with TemporaryDirectory() as temp:
        for world_size in sorted({case.world_size for case in selected}):
            group = [case for case in selected if case.world_size == world_size]
            runs = []
            for index, case in enumerate(group):
                root = Path(temp) / f"{world_size}-{index}"
                root.mkdir()
                config = write_run(
                    root,
                    by_name[case.architecture],
                    records=BATCHES_PER_EPOCH * world_size,
                )
                _vary_supervision(root / "features")
                output = root / "run"
                output.mkdir()
                overrides = [
                    f"output_dir={output}",
                    f"model.torch_dtype={case.dtype}",
                    f"deployment.trainer.nproc_per_node={world_size}",
                    f"training.fsdp_sharding={case.sharding}",
                    f"training.accumulation_steps={case.accumulation_steps}",
                    f"training.num_epochs={EPOCHS}",
                    # One more than the data holds: an extra update would show.
                    f"training.max_steps={case.updates + 1}",
                    f"training.total_steps={case.updates}",
                    "training.save_interval=1",
                    "training.log_interval=1",
                ]
                runs.append((str(config), overrides, str(output)))
            multiprocessing.spawn(
                _train_rank, args=(world_size, runs), nprocs=world_size, join=True
            )
            for case, (_, _, output) in zip(group, runs):
                recording[case.name] = _read_case(Path(output), case)
    return recording


def _same_bits(actual, expected):
    """Whether two tensors hold the same bytes: signed zeros and NaNs included."""
    return torch.equal(
        actual.contiguous().reshape(-1).view(torch.uint8),
        expected.contiguous().reshape(-1).view(torch.uint8),
    )


def differences(actual, expected, *, atol=0.0, where="recording"):
    """List where two recordings differ.

    Everything is compared exactly, bit for bit, except that floating-point
    tensors may differ by ``atol``: on an accelerator two runs of one revision
    agree only that far.
    """
    if type(actual) is not type(expected):
        return [f"{where}: {type(actual).__name__} != {type(expected).__name__}"]
    if isinstance(expected, dict):
        # 1, 1.0 and True are one key to a dict and three to a recording.
        typed = [{(type(key), key) for key in keys} for keys in (actual, expected)]
        if typed[0] != typed[1]:
            changed = sorted(str(key) for _, key in typed[0] ^ typed[1])
            return [f"{where}: keys differ in {changed}"]
        return [
            difference
            for key, value in expected.items()
            for difference in differences(
                actual[key], value, atol=atol, where=f"{where}.{key}"
            )
        ]
    if isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            return [f"{where}: {len(actual)} != {len(expected)} entries"]
        return [
            difference
            for index, value in enumerate(expected)
            for difference in differences(
                actual[index], value, atol=atol, where=f"{where}[{index}]"
            )
        ]
    if torch.is_tensor(expected):
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            return [f"{where}: dtype or shape differs"]
        if atol and expected.is_floating_point():
            same = torch.allclose(actual, expected, rtol=0, atol=atol, equal_nan=True)
        else:
            same = _same_bits(actual, expected)
        return [] if same else [f"{where}: values differ"]
    if isinstance(expected, float):
        same = struct.pack("<d", actual) == struct.pack("<d", expected)
    else:
        same = actual == expected
    return [] if same else [f"{where}: {actual!r} != {expected!r}"]


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    recorder = commands.add_parser("record", help="train every case and save it")
    recorder.add_argument("output")
    recorder.add_argument(
        "--every-case",
        action="store_true",
        help="also record the cases this host is not expected to place",
    )
    recorder.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=(),
        help="further world sizes to record under DDP and FSDP (implies --every-case)",
    )
    comparer = commands.add_parser("compare", help="compare two saved recordings")
    comparer.add_argument("actual")
    comparer.add_argument("expected")
    comparer.add_argument("--atol", type=float, default=0.0)
    options = parser.parse_args(arguments)
    if options.command == "record":
        selected = (
            every_case(options.ranks)
            if options.every_case or options.ranks
            else cases()
        )
        recording = record(selected)
        torch.save(recording, options.output)
        print(f"recorded {len(recording)} cases to {options.output}")
        return 0
    actual = torch.load(options.actual, weights_only=False)
    expected = torch.load(options.expected, weights_only=False)
    found = differences(actual, expected, atol=options.atol)
    for difference in found[:50]:
        print(difference)
    print(f"{len(expected)} cases, {len(found)} differences")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
