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

Use the same recorder, machine and torch build on both revisions. Recorder
changes can alter the stored fields and fixtures. By default the matrix runs
the topologies the host can place, including four-rank EAGLE3 sequence
parallelism when CUDA, FlashAttention and Yunchang are available.
"""

import argparse
import math
import os
import re
import struct
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
from torch.nn.modules.module import register_module_forward_pre_hook
from torch.nn.parallel import DistributedDataParallel
from torch.optim.optimizer import (
    register_optimizer_step_post_hook,
    register_optimizer_step_pre_hook,
)

from specforge.training.checkpoint import STATE_FILE
from specforge.utils import get_device_type
from tests.test_runtime.test_canonical_end_to_end import (
    drafts,
    write_features,
    write_run,
)

EPOCHS = 2
BATCHES_PER_EPOCH = 3
# What each case changes from one DDP rank that logs every update.
# Under bfloat16 the optimizer's fp32 masters hold bits the parameters do not.
# A log interval of two leaves updates whose forward collects no detailed
# metrics, which is what most updates of a real run are. An evaluated case
# runs an evaluation pass after every update.
_TWO_RANKS = dict(world_size=2, accumulation_steps=2)
_SHARDED = dict(sharding="SHARD_GRAD_OP")
TOPOLOGIES = (
    dict(),
    dict(accumulation_steps=2),
    dict(evaluated=True),
    _TWO_RANKS,
    _TWO_RANKS | dict(log_interval=2),
    _TWO_RANKS | dict(dtype="bfloat16"),
    _TWO_RANKS | dict(evaluated=True),
    _TWO_RANKS | _SHARDED,
    _TWO_RANKS | _SHARDED | dict(evaluated=True),
    dict(world_size=4, accumulation_steps=2, dtype="bfloat16") | _SHARDED,
)
# What a checkpoint's shared file says about the run's progress and weights.
SHARED_STATE = (
    "draft_state_dict",
    "replicated_optimizer_state",
    "global_step",
    "epoch",
    "epoch_batch",
    "epoch_samples",
    "world_size",
    "sampler_seed",
    "offline_sampler_version",
    "source_dataset_size",
)
# Half of a run's updates warm up: the shortest run has one warmup update.
WARMUP_RATIO = 0.5
CHECKPOINT = re.compile(r"specforge-run-step(\d+)")


@dataclass(frozen=True)
class Case:
    """One run: a draft architecture under one topology and accumulation window."""

    architecture: str
    world_size: int = 1
    accumulation_steps: int = 1
    sharding: str = "NO_SHARD"
    dtype: str = "float32"
    log_interval: int = 1
    evaluated: bool = False
    empty_eval_rank: int | None = None
    sp_ulysses_size: int = 1
    batch_size: int = 1

    @property
    def name(self):
        name = (
            f"{self.architecture}/ranks{self.world_size}"
            f"/accumulation{self.accumulation_steps}/{self.sharding}/{self.dtype}"
            f"/log{self.log_interval}"
        )
        if self.evaluated:
            name += "/evaluated"
        if self.empty_eval_rank is not None:
            name += f"/empty-eval-rank{self.empty_eval_rank}"
        if self.sp_ulysses_size > 1:
            name += f"/ulysses{self.sp_ulysses_size}/lk_lambda"
        if self.batch_size > 1:
            name += f"/batch{self.batch_size}"
        return name

    @property
    def batches_per_epoch(self):
        # SP peers share one sample. Each data replica therefore reads more
        # records from the same dataset of three records per global rank.
        return BATCHES_PER_EPOCH * self.sp_ulysses_size

    @property
    def window_size(self):
        return self.accumulation_steps * self.sp_ulysses_size

    @property
    def updates(self):
        """Optimizer updates the run reaches: windows span the epoch boundary."""
        return EPOCHS * self.batches_per_epoch // self.window_size


# One reference for each additional world size used by a bundled trainer recipe.
_RECIPE_CASES = (
    ("DFlash2DraftModel", 3),
    ("DFlashDraftModel", 7),
    ("Eagle3DraftModel", 8),
    ("MTPDraftModel", 10),
    ("DFlashDraftModel", 14),
)

_USP_CASE = Case(
    "Eagle3DraftModel",
    world_size=4,
    accumulation_steps=2,
    dtype="bfloat16",
    sp_ulysses_size=2,
)


def every_case(ranks=()):
    """Canonical runs, plus DDP and FSDP for every draft at any requested size."""
    topologies = list(TOPOLOGIES)
    for world_size in ranks:
        if world_size < 1:
            raise ValueError(f"--ranks requires positive world sizes, got {world_size}")
        wider = dict(world_size=world_size, accumulation_steps=2)
        topologies += [wider, wider | _SHARDED | dict(dtype="bfloat16")]
    selected = [
        Case(draft.architecture, **topology)
        for draft in drafts()
        for topology in topologies
    ]
    selected.extend(
        Case(
            architecture,
            world_size=world,
            accumulation_steps=2,
            dtype="bfloat16",
            **_SHARDED,
        )
        for architecture, world in _RECIPE_CASES
    )
    selected.append(_USP_CASE)
    selected.append(Case("DFlashDraftModel", accumulation_steps=2, batch_size=2))
    selected.append(
        Case("DFlashDraftModel", **_TWO_RANKS, evaluated=True, empty_eval_rank=1)
    )
    return list(dict.fromkeys(selected))


def _usp_available(device_type):
    if device_type != "cuda" or torch.version.hip is not None:
        return False
    if any(
        torch.cuda.get_device_capability(i)[0] < 8 for i in range(_USP_CASE.world_size)
    ):
        return False
    from specforge.modeling.draft.layers.attention.backends.cached_flash import (
        _std_flash_attn_varlen_func,
    )

    try:
        from yunchang.comm import SeqAllToAll4D
    except ImportError:
        return False
    return _std_flash_attn_varlen_func is not None and SeqAllToAll4D is not None


def cases(ranks=()):
    """The cases this host can place.

    An accelerator run needs one device per rank. FSDP refuses a CPU model
    unless the host has an accelerator to compute on, so a CPU trainer records
    the DDP cases only.
    """
    ranks = tuple(ranks)
    selected = every_case(ranks)
    device_type = get_device_type()
    if device_type == "cpu":
        return [
            case
            for case in selected
            if case.sharding == "NO_SHARD" and case.sp_ulysses_size == 1
        ]
    devices = torch.get_device_module(device_type).device_count()
    if any(world > devices for world in ranks):
        raise ValueError(
            f"--ranks {list(ranks)} requires more than the {devices} visible "
            f"{device_type} devices"
        )
    return [
        case
        for case in selected
        if case.world_size <= devices
        and (case.sp_ulysses_size == 1 or _usp_available(device_type))
    ]


def _vary_records(features):
    """Give the records different lengths and different supervised lengths.

    Ranks and micro-batches then hold different loss denominators, whether an
    objective counts supervised tokens or whole sequences, so a denominator
    pooled the wrong way changes the update.
    """
    for index, path in enumerate(sorted(features.iterdir())):
        record = torch.load(path)
        length = len(record["loss_mask"]) - index % 3
        record = {name: tensor[:length] for name, tensor in record.items()}
        supervised = record["loss_mask"].nonzero().flatten()
        record["loss_mask"][supervised[len(supervised) - index % 3 :]] = 0
        torch.save(record, path)


def _evaluation(root, case, draft):
    """Give the evaluation loader independent features and any rank-local shard."""
    held_out = root / "evaluation"
    write_features(held_out, draft, 2 * case.world_size)
    _vary_records(held_out)
    overrides = [f"data.eval_hidden_states_path={held_out}", "training.eval_interval=1"]
    rank_overrides = {}
    if case.empty_eval_rank is not None:
        # Offline sharding pads a nonempty manifest to equal lengths. Give this
        # DDP rank its own empty manifest so it joins just the eval reductions.
        empty = root / "empty-evaluation"
        empty.mkdir()
        rank_overrides[case.empty_eval_rank] = [f"data.eval_hidden_states_path={empty}"]
    return overrides, rank_overrides


@contextmanager
def _worker_environment(output, rank, world_size):
    """Give one local run a live rendezvous store, as a torchrun agent would."""
    address = "127.0.0.1"
    ports = dist.FileStore(str(Path(output) / "rendezvous"), world_size)
    if rank == 0:
        # Keep the server alive across the CLI call. All ranks, including zero,
        # connect as clients after the OS has bound the server's ephemeral port.
        server = dist.TCPStore(address, 0, is_master=True)
        ports.set("port", str(server.port))
    environment = {
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "NODE_RANK": "0",
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": address,
        "MASTER_PORT": ports.get("port").decode(),
        "TORCHELASTIC_USE_AGENT_STORE": "True",
    }
    with patch.dict(os.environ, environment):
        yield


def _on_host(value):
    """Snapshot nested tensors without changing numeric or sequence types."""
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
        self._optimizer = None
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

    def _after_step(self, optimizer, arguments, keywords):
        if self._model is None:
            return
        self.updates.append(
            {
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

    def _before_step(self, optimizer, arguments, keywords):
        self._optimizer = optimizer
        self.applied_lr.append([group["lr"] for group in optimizer.param_groups])

    def _keep_weights(self):
        if self._weights_due:
            update = self.updates[-1]
            update["weights"] = _on_host(self._model.state_dict())
            # The trainer's scheduler has advanced as well by now.
            update["optimizer"] = _on_host(self._optimizer.state_dict())
            update["next_lr"] = [group["lr"] for group in self._optimizer.param_groups]
            self._weights_due = False

    def ended(self):
        """Everything seen, once training has returned."""
        self._keep_weights()
        return {"applied_lr": self.applied_lr, "updates": self.updates}


def _train_rank(rank, world_size, runs, around_training):
    """Run every case on one rank, as a worker ``torchrun`` started would."""
    from specforge.cli import main

    # Ranks share this host's cores; one thread each keeps the sums in one order.
    torch.set_num_threads(1)
    for config, overrides, output, rank_overrides in runs:
        logged = []
        with (
            _worker_environment(output, rank, world_size),
            # The console logger every rank keeps when no tracker is configured.
            patch(
                "specforge.training.assembly._logger",
                lambda metrics, step: logged.append((step, _on_host(metrics))),
            ),
            _Replica() as replica,
            around_training() if around_training else nullcontext(),
        ):
            status = main(
                ["train", "-c", config, *overrides, *rank_overrides.get(rank, ())]
            )
            if status != 0:
                raise RuntimeError(f"specforge train exited with {status}: {config}")
            ended = {"logged": logged, "replica": replica.ended()}
        torch.save(ended, Path(output) / f"ended-rank{rank}.pt")


def _logged(events, case):
    """What the rank logged, by update: its training and its evaluation metrics."""
    trained, evaluated = {}, {}
    for step, values in events:
        if type(step) is not int:
            raise TypeError(
                f"{case.name}: logger step must be int, got {type(step).__name__}"
            )
        metrics = {
            name: value
            for name, value in values.items()
            if not name.startswith("perf/")
        }
        if any(name.startswith("eval/") for name in metrics):
            kind, entry = evaluated, metrics
        else:
            kind = trained
            entry = {
                # Computed on the host; the same on every device.
                "lr": metrics.pop("lr"),
                "metrics": metrics,
            }
        if step in kind:
            raise RuntimeError(f"{case.name}: a rank logged update {step} twice")
        kind[step] = entry
    updates = range(1, case.updates + 1)
    expected = (
        [step for step in updates if step % case.log_interval == 0],
        list(updates) if case.evaluated else [],
    )
    if (sorted(trained), sorted(evaluated)) != expected:
        raise RuntimeError(
            f"{case.name}: a rank logged updates {sorted(trained)} and "
            f"evaluations {sorted(evaluated)}"
        )
    return trained, evaluated


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
    trained, evaluated = zip(*(_logged(rank["logged"], case) for rank in ended))
    steps = sorted(
        int(match.group(1))
        for match in map(CHECKPOINT.fullmatch, os.listdir(output))
        if match
    )
    for rank in ended:
        applied = len(rank["replica"]["applied_lr"])
        if applied != len(steps):
            raise RuntimeError(
                f"{case.name}: checkpoints {steps} do not match the {applied} "
                "updates a rank applied"
            )
    return [
        {
            "checkpoint": step,
            # None at an update that is not logged, or not evaluated.
            "logged": [rank.get(step) for rank in trained],
            "evaluated": [rank.get(step) for rank in evaluated],
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


def record(selected, *, around_training=None):
    """Train every case and return ``{case name: what the run left}``.

    Cases of one world size share their worker processes, so a recording pays
    for the Python imports once per rank.

    Args:
        selected: The cases to train.
        around_training: A module-level function that returns a context
            manager; every rank enters it around each run. The conformance
            kit uses it to train a deliberately broken reference.
    """
    if not selected:
        raise ValueError("no case to record: this host can place none of them")
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
                    records=BATCHES_PER_EPOCH * world_size * case.batch_size,
                )
                _vary_records(root / "features")
                output = root / "run"
                output.mkdir()
                overrides, rank_overrides = [], {}
                if case.evaluated:
                    overrides, rank_overrides = _evaluation(
                        root, case, by_name[case.architecture]
                    )
                if case.sp_ulysses_size > 1:
                    overrides += [
                        "training.attention_backend=usp",
                        f"training.sp_ulysses_size={case.sp_ulysses_size}",
                        "training.objective_options.ttt_length=2",
                        "training.loss.terms={token: {kind: lk_lambda}}",
                    ]
                overrides += [
                    f"output_dir={output}",
                    f"model.torch_dtype={case.dtype}",
                    f"deployment.trainer.nproc_per_node={world_size}",
                    f"training.fsdp_sharding={case.sharding}",
                    f"training.batch_size={case.batch_size}",
                    f"training.accumulation_steps={case.accumulation_steps}",
                    f"training.num_epochs={EPOCHS}",
                    # One more than the data holds: an extra update would show.
                    f"training.max_steps={case.updates + 1}",
                    f"training.total_steps={case.updates}",
                    f"training.warmup_ratio={WARMUP_RATIO}",
                    "training.save_interval=1",
                    f"training.log_interval={case.log_interval}",
                ]
                runs.append((str(config), overrides, str(output), rank_overrides))
            multiprocessing.spawn(
                _train_rank,
                args=(world_size, runs, around_training),
                nprocs=world_size,
                join=True,
            )
            for case, (_, _, output, _) in zip(group, runs):
                recording[case.name] = _read_case(Path(output), case)
    return recording


def _same_bits(actual, expected):
    """Whether two tensors hold the same bytes: signed zeros and NaNs included."""
    return torch.equal(
        actual.contiguous().reshape(-1).view(torch.uint8),
        expected.contiguous().reshape(-1).view(torch.uint8),
    )


def differences(actual, expected, *, atol=0.0, where="recording", _metric_values=False):
    """List where two recordings differ.

    Floating tensors and device-computed metrics may differ by ``atol``.
    Other Python floats (including learning rates) and all types stay exact.
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
                actual[key],
                value,
                atol=atol,
                where=f"{where}.{key}",
                # The logger materializes device results as Python floats.
                # Only these two fields of the recording contain such metrics.
                _metric_values=_metric_values
                or (key == "evaluated" and "checkpoint" in expected)
                or (key == "metrics" and "lr" in expected),
            )
        ]
    if isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            return [f"{where}: {len(actual)} != {len(expected)} entries"]
        return [
            difference
            for index, value in enumerate(expected)
            for difference in differences(
                actual[index],
                value,
                atol=atol,
                where=f"{where}[{index}]",
                _metric_values=_metric_values,
            )
        ]
    if torch.is_tensor(expected):
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            return [f"{where}: dtype or shape differs"]
        if atol and expected.is_floating_point():
            # In the tensors' own dtype a bfloat16 threshold would round up.
            same = torch.allclose(
                actual.double(), expected.double(), rtol=0, atol=atol, equal_nan=True
            )
        else:
            same = _same_bits(actual, expected)
        return [] if same else [f"{where}: values differ"]
    if isinstance(expected, float):
        if _metric_values and atol:
            same = math.isclose(actual, expected, abs_tol=atol, rel_tol=0) or (
                math.isnan(actual) and math.isnan(expected)
            )
        else:
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
        help="further world sizes to record under DDP and FSDP",
    )
    comparer = commands.add_parser("compare", help="compare two saved recordings")
    comparer.add_argument("actual")
    comparer.add_argument("expected")
    comparer.add_argument("--atol", type=float, default=0.0)
    options = parser.parse_args(arguments)
    if options.command == "record":
        select = every_case if options.every_case else cases
        try:
            selected = select(options.ranks)
        except ValueError as exc:
            recorder.error(str(exc))
        recording = record(selected)
        torch.save(recording, options.output)
        print(f"recorded {len(recording)} cases to {options.output}")
        return 0
    actual = torch.load(options.actual, map_location="cpu", weights_only=True)
    expected = torch.load(options.expected, map_location="cpu", weights_only=True)
    found = differences(actual, expected, atol=options.atol)
    for difference in found[:50]:
        print(difference)
    print(f"{len(expected)} cases, {len(found)} differences")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
