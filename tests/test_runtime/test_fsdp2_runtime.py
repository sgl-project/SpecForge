"""The FSDP2 runtime: fp32 masters, bf16 compute, fp32 gradient accumulation."""

import copy
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from specforge.algorithms.objectives import build_objective_model
from specforge.modeling import build_draft_model
from specforge.optimizer import FP32Optimizer
from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.backends.fsdp2 import FSDP2TrainingBackend, _store_in_fp32
from specforge.training.contracts import PrecisionSpec
from specforge.training.program import program_spec
from specforge.training.runner import DataParallelRunner
from tests.test_algorithms.test_objectives import configurations, training_batch
from tests.test_runtime._fixtures import train_window

_GLOO = dist.is_available() and dist.is_gloo_available()
_FSDP2_BF16 = PrecisionSpec.resolve(runtime="fsdp2", model_dtype=torch.bfloat16)


def _optimizer_factory(module):
    return FP32Optimizer(
        module, lr=1e-3, max_grad_norm=1.0, total_steps=10, warmup_ratio=0.0
    )


def _program(dtype=torch.bfloat16):
    """A DFlash program built as ``build_bundle`` builds one: one dtype, and the
    bound tables and the teacher head declared frozen."""
    config = next(c for c in configurations() if c.architecture == "DFlashDraftModel")
    torch.manual_seed(0)
    draft = build_draft_model(config, dtype=dtype)
    bound = [draft.get_submodule(name) for name in config.target_bindings]
    for module in bound:
        module.load_target_weight(
            torch.randn(config.vocab_size, config.hidden_size, dtype=dtype)
        )
    width = training_batch(config)["target_last_hidden_states"].shape[-1]
    head = nn.Linear(width, config.vocab_size, bias=False, dtype=dtype)
    return build_objective_model(draft, target_head=head, frozen_modules=bound + [head])


def _batches(program, count, seed):
    torch.manual_seed(seed)
    config = program.draft_model.config.draft
    out = []
    for index in range(count):
        tensors = training_batch(config)
        out.append(TrainBatch([f"{seed}-{index}"], "dflash", tensors))
    return out


def _runner(program, sharding):
    backend = FSDP2TrainingBackend(
        ParallelConfig(sharding_strategy=sharding, fsdp_process_group=dist.group.WORLD),
        optimizer_factory=_optimizer_factory,
        precision=_FSDP2_BF16,
    )
    spec = program_spec(program)
    wrapped = backend.prepare_model(
        program,
        optimizer_target=program.draft_model,
        frozen=program.frozen,
        blocks=spec.partition.blocks,
    )
    return DataParallelRunner(wrapped, backend, spec), backend


def _worker(rank, world, store_path, out_dir, sharding):
    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=world
    )
    try:
        torch.manual_seed(0)
        program = _program()
        bound = {
            name: program.draft_model.get_submodule(name).weight.dtype
            for name in program.draft_model.config.draft.target_bindings
        }
        runner, backend = _runner(program, sharding)
        report = {"bound_dtypes": bound}
        report["trainable_dtypes"] = sorted(
            {str(p.dtype) for p in program.draft_model.parameters() if p.requires_grad}
        )
        report["bound_after"] = {
            name: program.draft_model.get_submodule(name).weight.dtype for name in bound
        }
        report["bound_is_plain"] = all(
            type(program.draft_model.get_submodule(name).weight) is nn.Parameter
            for name in bound
        )
        metrics = []
        for window in range(2):
            batches = _batches(program, 2, seed=100 * window + rank)
            metrics.append(train_window(runner, batches))
        report["losses"] = [m.loss for m in metrics]
        report["grad_norms"] = [float(m.grad_norm) for m in metrics]
        state = backend.state_dict()
        # What the generator draws next: a resumed run must draw the same.
        report["rng_draw"] = torch.rand(4)
        if rank == 0:
            report["model_dtypes"] = sorted(
                {
                    str(value.dtype)
                    for name, value in state["model"].items()
                    if name.startswith("draft_model.") and value.is_floating_point()
                }
            )
        torch.save(state, os.path.join(out_dir, f"state{rank}.pt"))
        # One more window, kept to compare with a resumed run.
        batches = _batches(program, 2, seed=300 + rank)
        report["next_loss"] = train_window(runner, batches).loss
        report["next_weights"] = {
            name: value.full_tensor().detach().clone()
            for name, value in program.draft_model.named_parameters()
            if value.requires_grad
        }
        torch.save(report, os.path.join(out_dir, f"report{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _resume_worker(rank, world, store_path, out_dir, sharding):
    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=world
    )
    try:
        torch.manual_seed(0)
        program = _program()
        saved = torch.load(os.path.join(out_dir, f"state{rank}.pt"), weights_only=False)
        shared = torch.load(os.path.join(out_dir, "state0.pt"), weights_only=False)
        # Draft weights load before the wrap, as the trainer does; the masters
        # come back with this rank's optimizer state.
        program.load_state_dict(shared["model"], strict=False)
        runner, backend = _runner(program, sharding)
        backend.load_state_dict({"optimizer": saved["optimizer"], "rng": saved["rng"]})
        report = {"rng_draw": torch.rand(4)}
        batches = _batches(program, 2, seed=300 + rank)
        report["next_loss"] = train_window(runner, batches).loss
        report["next_weights"] = {
            name: value.full_tensor().detach().clone()
            for name, value in program.draft_model.named_parameters()
            if value.requires_grad
        }
        torch.save(report, os.path.join(out_dir, f"resumed{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _on_host_fsdp1():
    """FSDP1 computes on the host only when told so."""
    from unittest.mock import patch

    from torch.distributed.fsdp import FullyShardedDataParallel

    original = FullyShardedDataParallel.__init__

    def on_host(self, module, *args, **kwargs):
        kwargs.setdefault("device_id", torch.device("cpu"))
        original(self, module, *args, **kwargs)

    return patch.object(FullyShardedDataParallel, "__init__", on_host)


def _reference_backend(sharding):
    from specforge.optimizer import BF16Optimizer

    return FSDPTrainingBackend(
        ParallelConfig(sharding_strategy=sharding, fsdp_process_group=dist.group.WORLD),
        precision=PrecisionSpec.resolve(runtime="fsdp", model_dtype=torch.bfloat16),
        optimizer_factory=lambda module: BF16Optimizer(
            module, lr=1e-3, max_grad_norm=1.0, total_steps=10, warmup_ratio=0.0
        ),
    )


def _written(backend, shapes):
    """What the reference wrote, whole, from the writer's own shards.

    Read from the FSDP1 wrap that wrote the state, where the code under test
    replays that wrap elsewhere. ``shapes``: the trainable parameters' full
    shapes, taken before the wrap. Rank 0 returns, per trainable parameter, its
    fp32 master and, if the reference stepped it, its step and moments.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    optimizer = backend.optimizer
    spans = {}
    for unit in backend.module.modules():
        if isinstance(unit, FSDP) and unit._handle is not None:
            flat = unit._handle.flat_param
            for param, info in zip(flat._params, flat._shard_param_infos):
                spans[id(param)] = (
                    (info.intra_param_start_idx, info.intra_param_end_idx + 1)
                    if info.in_shard
                    else None
                )
    mine = []
    for param, master in zip(optimizer.model_params, optimizer.fp32_params):
        span = spans.get(id(param), (0, param.numel())) if spans else (0, param.numel())
        state = optimizer.optimizer.state.get(master, {})
        mine.append((tuple(param.shape) if not spans else None, span, master, state))
    everyone = [None] * dist.get_world_size()
    dist.all_gather_object(everyone, mine, group=dist.group.WORLD)
    if dist.get_rank() != 0:
        return None
    whole = []
    for index, shape in enumerate(shapes):
        numel = int(torch.Size(shape).numel())
        master = torch.zeros(numel)
        moments = {name: torch.zeros(numel) for name in ("exp_avg", "exp_avg_sq")}
        steps = set()
        for rank_parts in everyone:
            _, span, values, state = rank_parts[index]
            if span is None:
                continue
            master[span[0] : span[1]] = values.reshape(-1)
            if state:
                steps.add(float(state["step"]))
                for name in moments:
                    moments[name][span[0] : span[1]] = state[name].reshape(-1)
        whole.append(
            {
                "master": master.view(shape),
                "step": steps.pop() if steps else None,
                **{name: value.view(shape) for name, value in moments.items()},
            }
        )
    holds_nothing = any(
        rank_parts[index][1] is None
        for rank_parts in everyone
        for index in range(len(shapes))
    )
    return {"whole": whole, "some_rank_holds_no_slice": holds_nothing}


def _reference_worker(rank, world, store_path, out_dir, sharding):
    """Train with the reference runtime and write its optimizer state."""
    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=world
    )
    try:
        torch.manual_seed(0)
        program = _program()
        shapes = [p.shape for p in program.draft_model.parameters() if p.requires_grad]
        backend = _reference_backend(sharding)
        spec = program_spec(program)
        with _on_host_fsdp1():
            wrapped = backend.prepare_model(
                program,
                optimizer_target=program.draft_model,
                frozen=program.frozen,
                blocks=spec.partition.blocks,
            )
        runner = DataParallelRunner(wrapped, backend, spec)
        for window in range(2):
            train_window(runner, _batches(program, 2, seed=50 * window + rank))
        torch.save(
            {
                "saved": backend.optimizer.state_dict(),
                "written": _written(backend, shapes),
            },
            os.path.join(out_dir, f"reference{rank}.pt"),
        )
    finally:
        dist.destroy_process_group()


def _read_reference_worker(rank, world, store_path, out_dir, sharding):
    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=world
    )
    try:
        torch.manual_seed(0)
        program = _program()
        reference = torch.load(
            os.path.join(out_dir, f"reference{rank}.pt"), weights_only=False
        )
        written = torch.load(
            os.path.join(out_dir, "reference0.pt"), weights_only=False
        )["written"]
        runner, backend = _runner(program, "SHARD_GRAD_OP")
        backend.load_state_dict({"optimizer": reference["saved"]})
        trainable = [p for p in program.draft_model.parameters() if p.requires_grad]
        report = {"some_rank_holds_no_slice": written["some_rank_holds_no_slice"]}
        report["mismatches"] = _mismatches(backend, trainable, written["whole"])
        batches = _batches(program, 2, seed=7 + rank)
        report["loss"] = train_window(runner, batches).loss
        torch.save(report, os.path.join(out_dir, f"read{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _mismatches(backend, trainable, whole):
    """Each trainable parameter whose restored master or state differs."""
    wrong = []
    for index, (param, expected) in enumerate(zip(trainable, whole)):
        state = backend.optimizer.optimizer.state.get(param, {})
        same = torch.equal(param.full_tensor(), expected["master"])
        if expected["step"] is None:
            same = same and not state
        else:
            same = (
                same
                and float(state["step"]) == expected["step"]
                and torch.equal(state["exp_avg"].full_tensor(), expected["exp_avg"])
                and torch.equal(
                    state["exp_avg_sq"].full_tensor(), expected["exp_avg_sq"]
                )
            )
        if not same:
            wrong.append(index)
    return wrong


class _Pair(nn.Module):
    """A trainable vector one rank's FSDP1 shard holds whole, beside a frozen
    one in the same unit: that rank's masters look like DDP's."""

    def __init__(self):
        super().__init__()
        self.vector = nn.Parameter(torch.randn(8))
        self.fixed = nn.Parameter(torch.randn(8), requires_grad=False)

    def forward(self, x):
        return (self.vector * self.fixed * x).sum()


def _pair_worker(rank, world, store_path, out_dir, _sharding):
    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=world
    )
    try:
        torch.manual_seed(0)
        program = _Pair().to(torch.bfloat16)
        shape = program.vector.shape
        backend = _reference_backend("SHARD_GRAD_OP")
        with _on_host_fsdp1():
            backend.prepare_model(program)
        x = torch.full((8,), 1.0 + rank, dtype=torch.bfloat16)
        backend.backward(backend.module(x).float())
        backend.optimizer.step()
        saved = backend.optimizer.state_dict()
        written = [_written(backend, [shape])]
        dist.broadcast_object_list(written, src=0)
        torch.manual_seed(0)
        again = _Pair().to(torch.bfloat16)
        resumed = _toy_backend("SHARD_GRAD_OP")
        resumed.prepare_model(again)
        resumed.load_state_dict({"optimizer": saved})
        report = {
            "whole_master_here": tuple(saved["fp32_params"][0].shape) == (8,),
            "mismatches": _mismatches(resumed, [again.vector], written[0]["whole"]),
        }
        resumed.backward(again(x).float())
        resumed.optimizer.step()
        torch.save(report, os.path.join(out_dir, f"pair{rank}.pt"))
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(_GLOO, "needs gloo")
class FSDP2RuntimeTest(unittest.TestCase):
    def _run(self, worker, directory, sharding, world=2):
        store = os.path.join(directory, f"store-{worker.__name__}")
        mp.spawn(worker, args=(world, store, directory, sharding), nprocs=world)

    def test_two_ranks_train_resume_and_write_the_draft_in_its_dtype(self):
        for sharding in ("SHARD_GRAD_OP", "NO_SHARD"):
            with self.subTest(sharding=sharding), tempfile.TemporaryDirectory() as d:
                self._run(_worker, d, sharding)
                report = torch.load(os.path.join(d, "report0.pt"), weights_only=False)
                # The trainable draft is fp32; bound target tables stay as built,
                # replicated, outside FSDP.
                self.assertEqual(report["trainable_dtypes"], ["torch.float32"])
                self.assertEqual(report["bound_after"], report["bound_dtypes"])
                self.assertTrue(report["bound_is_plain"])
                # The checkpoint's draft weights keep the draft's dtype.
                self.assertEqual(report["model_dtypes"], ["torch.bfloat16"])
                self.assertTrue(all(torch.isfinite(torch.tensor(report["losses"]))))
                # A resumed run continues bit for bit.
                self._run(_resume_worker, d, sharding)
                for rank in range(2):
                    kept = torch.load(
                        os.path.join(d, f"report{rank}.pt"), weights_only=False
                    )
                    resumed = torch.load(
                        os.path.join(d, f"resumed{rank}.pt"), weights_only=False
                    )
                    self.assertTrue(torch.equal(resumed["rng_draw"], kept["rng_draw"]))
                    self.assertEqual(resumed["next_loss"], kept["next_loss"])
                    for name, value in kept["next_weights"].items():
                        self.assertTrue(
                            torch.equal(resumed["next_weights"][name], value), name
                        )

    def test_reads_the_reference_runtimes_optimizer_state_exactly(self):
        for sharding in ("SHARD_GRAD_OP", "NO_SHARD"):
            with self.subTest(sharding=sharding), tempfile.TemporaryDirectory() as d:
                self._run(_reference_worker, d, sharding)
                self._run(_read_reference_worker, d, sharding)
                for rank in range(2):
                    read = torch.load(
                        os.path.join(d, f"read{rank}.pt"), weights_only=False
                    )
                    self.assertEqual(read["mismatches"], [])
                    self.assertTrue(torch.isfinite(torch.tensor(read["loss"])))
                # Under FSDP1 some rank holds no slice of some parameter, and
                # so no optimizer state for it.
                self.assertEqual(
                    read["some_rank_holds_no_slice"], sharding == "SHARD_GRAD_OP"
                )

    def test_ranks_agree_on_a_layout_one_rank_cannot_tell(self):
        with tempfile.TemporaryDirectory() as d:
            self._run(_pair_worker, d, "SHARD_GRAD_OP")
            reports = [
                torch.load(os.path.join(d, f"pair{rank}.pt"), weights_only=False)
                for rank in range(2)
            ]
            self.assertEqual([r["whole_master_here"] for r in reports], [True, False])
            for report in reports:
                self.assertEqual(report["mismatches"], [])


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 5, bias=False)  # five rows: uneven on two ranks
        self.table = nn.Embedding(3, 4)  # frozen, inside a sharded block
        self.table.weight.requires_grad_(False)

    def forward(self, x):
        return self.proj(x + self.table.weight.sum(0))


class _Uneven(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _Block()
        self.head = nn.Linear(5, 3, bias=False)

    def forward(self, x):
        return self.head(self.block(x))


def _toy_backend(sharding, group=None):
    return FSDP2TrainingBackend(
        ParallelConfig(
            sharding_strategy=sharding, fsdp_process_group=group or dist.group.WORLD
        ),
        optimizer_factory=_optimizer_factory,
        precision=_FSDP2_BF16,
    )


def _wrapped_toy(sharding, weights=None):
    program = _Uneven().to(torch.bfloat16)
    if weights is not None:
        program.load_state_dict(weights)
    backend = _toy_backend(sharding)
    backend.prepare_model(program, frozen=(program.block.table,), blocks=("_Block",))
    return program, backend


def _toy_step(program, backend, x):
    backend.backward(program(x).float().pow(2).mean(), is_boundary=True)
    backend.optimizer.step()


def _full(program):
    return {
        name: value.full_tensor().detach().clone()
        for name, value in program.named_parameters()
        if value.requires_grad
    }


def _toy_worker(rank, world, store_path, out_dir, sharding):
    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=world
    )
    try:
        # Each rank builds its own initialization: NO_SHARD must replace it with
        # rank 0's, as DDP does.
        torch.manual_seed(rank)
        program, backend = _wrapped_toy(sharding)
        report = {
            "table_is_plain": type(program.block.table.weight) is nn.Parameter,
            "replicated": backend.optimizer_state_is_replicated,
            "local": {
                name: value.to_local().detach().clone()
                for name, value in program.named_parameters()
                if value.requires_grad
            },
        }
        inputs = torch.randn(2, 2, 4, generator=torch.Generator().manual_seed(7))
        _toy_step(program, backend, inputs[0].to(torch.bfloat16))
        weights = {
            name: value.to(torch.bfloat16) for name, value in _full(program).items()
        }
        weights["block.table.weight"] = program.block.table.weight.detach().clone()
        # Copied as a checkpoint copies it: the live state keeps training.
        optimizer = copy.deepcopy(backend.optimizer.state_dict())
        _toy_step(program, backend, inputs[1].to(torch.bfloat16))
        report["continued"] = _full(program)
        # A new program from the checkpoint's weights and this rank's optimizer
        # state continues exactly, uneven shards included.
        again, resumed = _wrapped_toy(sharding, weights)
        resumed.optimizer.load_state_dict(optimizer)
        _toy_step(again, resumed, inputs[1].to(torch.bfloat16))
        report["resumed"] = _full(again)
        if sharding == "NO_SHARD":
            # Replicating over part of the job is refused.
            part = dist.new_group([0])
            if rank == 0:
                try:
                    _toy_backend(sharding, part)._mesh(torch.device("cpu"))
                except ValueError as error:
                    report["part_refused"] = str(error)
        torch.save(report, os.path.join(out_dir, f"toy{rank}.pt"))
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(_GLOO, "needs gloo")
class FSDP2LayoutTest(unittest.TestCase):
    def test_uneven_shards_resume_frozen_blocks_and_aligned_replicas(self):
        for sharding in ("SHARD_GRAD_OP", "NO_SHARD"):
            with self.subTest(sharding=sharding), tempfile.TemporaryDirectory() as d:
                store = os.path.join(d, "store")
                mp.spawn(_toy_worker, args=(2, store, d, sharding), nprocs=2)
                reports = [
                    torch.load(os.path.join(d, f"toy{rank}.pt"), weights_only=False)
                    for rank in range(2)
                ]
                for report in reports:
                    self.assertTrue(report["table_is_plain"])
                    self.assertEqual(report["replicated"], sharding == "NO_SHARD")
                    for name, value in report["continued"].items():
                        self.assertTrue(torch.equal(report["resumed"][name], value), name)
                if sharding == "NO_SHARD":
                    for name, value in reports[0]["local"].items():
                        self.assertTrue(torch.equal(reports[1]["local"][name], value))
                    self.assertIn("every rank of the job", reports[0]["part_refused"])


class _Aliased(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
        self.b = self.a
        self.register_buffer("codes", torch.ones(2, dtype=torch.bfloat16))
        self.register_buffer(
            "cache", torch.ones(2, dtype=torch.bfloat16), persistent=False
        )
        self.frozen = nn.Linear(2, 2).to(torch.bfloat16).requires_grad_(False)


class StoreInFP32Test(unittest.TestCase):
    def test_aliases_stay_one_tensor_and_buffers_are_written_as_built(self):
        module = _Aliased()
        module.frozen.register_buffer("scale", torch.ones(2, dtype=torch.bfloat16))
        written = _store_in_fp32(
            module, frozen=(module.frozen,), artifact_dtype=torch.float16
        )
        self.assertIs(module.a, module.b)
        self.assertEqual(module.a.dtype, torch.float32)
        self.assertEqual(module.codes.dtype, torch.float32)
        self.assertEqual(module.cache.dtype, torch.float32)
        self.assertEqual(module.frozen.weight.dtype, torch.bfloat16)
        self.assertEqual(module.frozen.scale.dtype, torch.bfloat16)
        # Parameters leave in the artifact dtype, buffers as they were built.
        self.assertEqual(
            written,
            {"a": torch.float16, "b": torch.float16, "codes": torch.bfloat16},
        )


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(64, 64, bias=False)

    def forward(self, x):
        return self.layer(x)


@unittest.skipUnless(_GLOO, "needs gloo")
class AccumulationPrecisionTest(unittest.TestCase):
    """Micro-batch gradients accumulate in fp32, not in the compute dtype."""

    def test_window_gradient_is_the_fp32_sum_of_the_bf16_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group(
                "gloo",
                init_method=f"file://{os.path.join(directory, 'store')}",
                rank=0,
                world_size=1,
            )
            try:
                torch.manual_seed(0)
                reference = _Toy().to(torch.bfloat16)
                program = _Toy().to(torch.bfloat16)
                program.load_state_dict(reference.state_dict())
                backend = FSDP2TrainingBackend(
                    ParallelConfig(
                        sharding_strategy="SHARD_GRAD_OP",
                        fsdp_process_group=dist.group.WORLD,
                    ),
                    precision=_FSDP2_BF16,
                )
                backend.prepare_model(program)
                micro_batches = [
                    torch.randn(8, 64, dtype=torch.bfloat16) * (1 + 3 * index)
                    for index in range(32)
                ]
                exact = torch.zeros(64, 64, dtype=torch.float64)
                in_bf16 = torch.zeros(64, 64, dtype=torch.bfloat16)
                for index, x in enumerate(micro_batches):
                    # The reference computes each gradient in bf16, as the
                    # runtime does, then sums exactly and in bf16.
                    reference.zero_grad()
                    reference(x).float().pow(2).mean().backward()
                    exact += reference.layer.weight.grad.double()
                    in_bf16 += reference.layer.weight.grad
                    backend.backward(
                        program(x).float().pow(2).mean(),
                        is_boundary=index == len(micro_batches) - 1,
                    )
                grad = program.layer.weight.grad.full_tensor()
                self.assertEqual(grad.dtype, torch.float32)
                fp32_error = (grad.double() - exact).abs().max() / exact.abs().max()
                bf16_error = (in_bf16.double() - exact).abs().max() / exact.abs().max()
                self.assertLess(float(fp32_error), 1e-6)
                self.assertGreater(float(bf16_error), 100 * float(fp32_error))
            finally:
                dist.destroy_process_group()


class FSDP2PrecisionTest(unittest.TestCase):
    def test_refuses_a_precision_it_does_not_hold(self):
        reference = PrecisionSpec.resolve(runtime="fsdp", model_dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "fp32 parameters as the masters"):
            FSDP2TrainingBackend(ParallelConfig(), precision=reference)


class RuntimeStateTest(unittest.TestCase):
    def test_fsdp2_loads_no_model_state(self):
        backend = FSDP2TrainingBackend(ParallelConfig(), precision=_FSDP2_BF16)
        with self.assertRaisesRegex(ValueError, "before the wrap"):
            backend.load_state_dict({"model": {}})

    def test_each_runtime_refuses_the_other_s_optimizer_state(self):
        with self.assertRaisesRegex(ValueError, "not written by the fsdp2 runtime"):
            FP32Optimizer(nn.Linear(2, 2), lr=1e-3).load_state_dict(
                {"lr_scheduler_type": "cosine"}
            )
        reference = FSDPTrainingBackend(
            ParallelConfig(),
            precision=PrecisionSpec.resolve(runtime="fsdp", model_dtype=torch.bfloat16),
        )
        with self.assertRaisesRegex(ValueError, "resume it with training.runtime=fsdp2"):
            reference.load_state_dict({"optimizer": {FP32Optimizer.FORMAT_KEY: True}})


class FP32OptimizerTest(unittest.TestCase):
    def test_refuses_low_precision_parameters(self):
        with self.assertRaisesRegex(ValueError, "fp32 parameters"):
            FP32Optimizer(nn.Linear(2, 2).to(torch.bfloat16), lr=1e-3)

    def test_step_clips_checks_and_clears(self):
        module = nn.Linear(4, 4)
        optimizer = FP32Optimizer(module, lr=1e-2, max_grad_norm=0.1, total_steps=5)
        module(torch.randn(3, 4)).pow(2).sum().backward()
        norm = torch.cat([p.grad.flatten() for p in module.parameters()]).norm()
        returned = optimizer.step()
        self.assertAlmostEqual(float(returned), float(norm), places=5)
        self.assertTrue(all(p.grad is None for p in module.parameters()))
        module(torch.randn(3, 4)).pow(2).sum().backward()
        with self.assertRaisesRegex(ValueError, "denominator"):
            optimizer.step(loss_denominator=torch.tensor(0.0))
        self.assertTrue(all(p.grad is None for p in module.parameters()))


if __name__ == "__main__":
    unittest.main()
