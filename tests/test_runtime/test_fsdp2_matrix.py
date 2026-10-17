# coding=utf-8
"""FSDP2 coverage matrix (2 GPUs): every algorithm x sharding mode, plus the
behaviors a parity number cannot see.

Grid: {eagle3, peagle, dflash, dflash2, domino, dspark, mtp} x
{SHARD_GRAD_OP, FULL_SHARD, NO_SHARD}.  Each cell trains the same tiny model
with DISTINCT per-rank micro-batches under FSDP1 and FSDP2 and compares loss,
grad norm and weights within the bf16 reduce tolerance; NO_SHARD is DDP under
both implementations and must be bitwise identical.

Beyond the grid:
  * accumulation: FSDP2's gradient-sync toggle flips exactly acc-1 times per
    window and parameters stay resident (unsharded) across the window under
    SHARD_GRAD_OP but not under FULL_SHARD;
  * collective schedule: per micro-step all-gather / reduce-scatter / all-reduce
    counts equal FSDP1's in both modes (root keeps params for backward);
  * NO_SHARD: neither backend reduces DDP gradients before the boundary
    (``micro_step()`` covers the forward, where DDP decides);
  * unequal valid-token counts per rank: the normalized pre-clip gradients and
    their global norm match a merged-batch reference, and a no-op
    ``scale_gradients`` is caught (AdamW + clipping would hide it in weights);
  * launch-level resume: ``build_offline_runtime`` -> ``fit`` -> checkpoint ->
    new runtime with ``resume_from`` -> ``fit`` equals an uninterrupted run,
    for fsdp1 and fsdp2;
  * MTP keeps its tied frozen tables as one parameter, P-EAGLE's [1, 1, H]
    parameter shards to zero rows on rank 1, DFlash2's fused Triton head stays
    eligible after the FSDP2 wrap.

CPU: a shared recipe with ``fsdp_version: 2`` still loads for the CPU producer.
Run on the H200 box via rcli (>= 2 GPUs); the GPU tests self-skip otherwise.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

CUDA = torch.cuda.is_available()
NGPU = torch.cuda.device_count() if CUDA else 0
WORLD = 2
BS = 2
ACC = 2
STEPS = 3
ALGORITHMS = ("eagle3", "peagle", "dflash", "dflash2", "domino", "dspark", "mtp")
SHARDINGS = ("SHARD_GRAD_OP", "FULL_SHARD", "NO_SHARD")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _optimizer_factory(module):
    from specforge.optimizer import BF16Optimizer

    return BF16Optimizer(
        module, lr=1e-3, max_grad_norm=0.5, warmup_ratio=0.0, total_steps=100
    )


def _cpu_sd(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _max_diff(a, b):
    if sorted(a) != sorted(b):
        return float("inf")
    worst = 0.0
    for key in a:
        x, y = a[key].cpu(), b[key].cpu()
        if x.is_floating_point():
            worst = max(worst, float((x.float() - y.float()).abs().max()))
        elif not torch.equal(x, y):
            worst = float("inf")
    return worst


def _rel(xs, ys):
    return max(abs(x - y) / max(abs(y), 1e-8) for x, y in zip(xs, ys))


class SharedRecipeTest(unittest.TestCase):
    def test_fsdp2_recipe_loads_for_cpu_producer_and_consumer(self):
        from specforge.cli import _config_for_role
        from specforge.config import Config
        from specforge.config.schema import apply_overrides

        # any recipe becomes an fsdp2 recipe with one override
        path = os.path.join(
            REPO_ROOT,
            "examples/configs/online/disaggregated/external",
            "qwen3-4b-dflash-online.yaml",
        )
        with mock.patch.dict("os.environ", {"SPECFORGE_DEVICE": "cpu"}):
            cfg = apply_overrides(Config.from_file(path), ["training.fsdp_version=2"])
            self.assertEqual(cfg.training.fsdp_version, 2)
            # the capture-only producer never wraps a model; the shared recipe must not be rejected on its CPU host
            producer = _config_for_role(cfg, "producer")
            consumer = _config_for_role(cfg, "consumer")
        self.assertEqual(producer.training.role, "producer")
        self.assertEqual(consumer.training.fsdp_version, 2)


# ---- toy composite for the unequal-token-count check (loss_terms path) ----
class _ToyBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.fc = nn.Linear(h, h, bias=False)
        self.norm_w = nn.Parameter(torch.ones(h))

    def forward(self, x):
        return F.silu(self.fc(x)) * self.norm_w + x


class _ToyDraft(nn.Module):
    _no_split_modules = ["_ToyBlock"]

    def __init__(self, h, n=2):
        super().__init__()
        self.fc_in = nn.Linear(h, h)
        self.layers = nn.ModuleList([_ToyBlock(h) for _ in range(n)])

    def forward(self, x):
        x = self.fc_in(x)
        for layer in self.layers:
            x = layer(x)
        return x


class _ToyComposite(nn.Module):
    def __init__(self, v=128, h=64):
        super().__init__()
        self.embed_tokens = nn.Embedding(v, h)
        self.embed_tokens.weight.requires_grad_(False)
        self.lm_head = nn.Linear(h, v, bias=False)
        self.lm_head.weight.requires_grad_(False)
        self.draft_model = _ToyDraft(h)

    def forward(self, ids, labels, loss_mask):
        logits = F.linear(self.draft_model(self.embed_tokens(ids)), self.lm_head.weight)
        per_token = F.cross_entropy(
            logits.float().view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
        )
        mask = loss_mask.view(-1).float()
        return (per_token * mask).sum(), mask.sum()


def _toy_strategy_class():
    from specforge.training.strategies.base import DraftTrainStrategy, StepOutput

    class TokenSumStrategy(DraftTrainStrategy):
        name = "toy"
        required_features = {"ids", "labels", "loss_mask"}

        def __init__(self, model):
            self.model = model

        def trainable_module(self):
            return self.model

        def forward_loss(self, batch, ctx=None):
            device = next(self.model.parameters()).device
            t = batch.tensors
            numerator, denominator = self.model(
                t["ids"].to(device), t["labels"].to(device), t["loss_mask"].to(device)
            )
            # the trainer backpropagates the additive numerator and divides by the GLOBAL token count across ranks and accumulation steps
            return StepOutput(
                loss=numerator / denominator.clamp_min(1),
                metrics={},
                loss_terms=(numerator, denominator),
            )

    return TokenSumStrategy


# ---- the 2-GPU worker ----
def _algorithms(rank, workdir):
    """Return {name: (fresh_model, make_strategy, this_rank_batch)}."""
    from specforge.algorithms.eagle3.model import OnlineEagle3Model
    from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
    from specforge.modeling.target.target_head import TargetHead
    from specforge.training.strategies.base import (
        DFlashTrainStrategy,
        DominoTrainStrategy,
        DSparkTrainStrategy,
        Eagle3TrainStrategy,
        MTPTrainStrategy,
        PEagleTrainStrategy,
    )
    from tests.test_runtime import _fixtures as fx

    rank_dir = os.path.join(workdir, f"rank{rank}")
    os.makedirs(rank_dir, exist_ok=True)
    seed = 100 + rank  # DISTINCT data per rank
    draft_config = AutoDraftModelConfig.from_file(os.path.join(workdir, "draft.json"))
    head = TargetHead.from_pretrained(
        os.path.join(workdir, "target"), lm_head_key="lm_head.weight"
    )

    def fresh_eagle3():
        torch.manual_seed(0)
        dm = AutoDraftModel.from_config(
            draft_config, attention_backend="flex_attention", torch_dtype=torch.bfloat16
        ).cuda()
        dm.load_vocab_mapping(os.path.join(workdir, "vm.pt"))
        dm.freeze_embedding()
        return OnlineEagle3Model(
            draft_model=dm, length=3, attention_backend="flex_attention"
        ).cuda()

    eagle3_batch = next(
        iter(
            fx.build_offline_eagle3_loader(
                os.path.join(workdir, "features"),
                batch_size=BS,
                run_id="fsdp2-matrix",
                ttt_length=3,
                max_len=512,
                ref_slice=slice(rank * BS, (rank + 1) * BS),
            )
        )
    )

    def seeded(builder):
        def fresh():
            torch.manual_seed(0)
            return builder()

        return fresh

    family = dict(hidden=64, block_size=4, num_anchors=8, attention_backend="sdpa")
    return {
        "eagle3": (
            fresh_eagle3,
            lambda w: Eagle3TrainStrategy(w, target_head=head),
            eagle3_batch,
        ),
        "peagle": (
            seeded(lambda: fx.build_peagle(os.path.join(rank_dir, "peagle"))),
            PEagleTrainStrategy,
            fx.peagle_batch(seed),
        ),
        "dflash": (
            seeded(lambda: fx.build_dflash(os.path.join(rank_dir, "dflash"), **family)[0]),
            DFlashTrainStrategy,
            fx.dflash_family_batch("dflash", seed, width=64),
        ),
        "dflash2": (
            seeded(lambda: fx.build_dflash2()[0]),
            DFlashTrainStrategy,
            fx.dflash_family_batch("dflash", seed, width=128, seq=48, vocab=500),
        ),
        "domino": (
            seeded(lambda: fx.build_domino(os.path.join(rank_dir, "domino"), **family)[0]),
            DominoTrainStrategy,
            fx.dflash_family_batch("domino", seed, width=64),
        ),
        "dspark": (
            seeded(
                lambda: fx.build_dspark(
                    os.path.join(rank_dir, "dspark"),
                    hidden=64,
                    block_size=4,
                    attention_backend="sdpa",
                )[0]
            ),
            DSparkTrainStrategy,
            fx.dflash_family_batch("dspark", seed, width=64, with_target_last=True),
        ),
        "mtp": (seeded(fx.build_mtp), MTPTrainStrategy, fx.mtp_batch(seed)),
    }


def _matrix_worker(rank, world_size, workdir, results_dir):
    import torch.distributed as dist
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor

    from tests.test_runtime import _fixtures as fx

    fx.init_rank_distributed(rank, world_size, port="29594")
    torch.use_deterministic_algorithms(True, warn_only=True)

    from specforge.training.backend import (
        FSDP1TrainingBackend,
        FSDP2TrainingBackend,
        ParallelConfig,
    )
    from specforge.training.controller import TrainerCore
    from specforge.training.strategies.base import StepContext

    results = {"matrix": {}, "accum": {}, "tokens": {}, "launch_resume": {}, "comm": {}, "ddp_nosync": {}}

    def train(backend_cls, sharding, model, make_strategy, batch, *, wrap=True,
              steps=STEPS, acc=ACC, observe=None, before_step=None, no_scale=False,
              on_ready=None):
        pc = ParallelConfig.from_distributed()
        pc.sharding_strategy = sharding
        pc.fsdp_version = 2 if backend_cls is FSDP2TrainingBackend else 1
        backend = backend_cls(pc, optimizer_factory=_optimizer_factory)
        wrapped = backend.prepare_model(model, wrap=wrap, optimizer_target=model.draft_model)
        if no_scale:
            backend.scale_gradients = lambda factor: None  # mutation: drop the denominator
        if before_step is not None:
            orig_step = backend.optimizer.step

            def step(**kw):
                before_step(wrapped, backend)  # sees normalized, pre-clip gradients
                return orig_step(**kw)

            backend.optimizer.step = step
        core = TrainerCore(make_strategy(wrapped), backend, accumulation_steps=acc)
        if on_ready is not None:
            on_ready(wrapped, backend)  # after the wrap, before the first micro-step
        losses, gnorms = [], []
        for step in range(steps):
            ctx = StepContext(global_step=step, total_steps=steps)
            for micro in range(acc):
                result = core.train_step(batch, ctx)
                if observe is not None:
                    observe(wrapped, step, micro, result.optimizer_stepped)
            losses.append(float(result.loss))
            gnorms.append(float(result.grad_norm))
        return backend, wrapped, losses, gnorms

    # ---- the grid ------------------------------------------------------------
    algorithms = _algorithms(rank, workdir)
    fused_probe = torch.zeros(4, 64, device="cuda", dtype=torch.bfloat16)
    for name, (fresh, make_strategy, batch) in algorithms.items():
        gold = _cpu_sd(fresh())

        def make(fresh=fresh, gold=gold):
            model = fresh()
            model.load_state_dict(gold)
            return model

        for sharding in SHARDINGS:
            m1 = make()
            b1, w1, l1, g1 = train(FSDP1TrainingBackend, sharding, m1, make_strategy, batch)
            full1 = b1.state_dict()["model"]
            m2 = make()
            b2, w2, l2, g2 = train(FSDP2TrainingBackend, sharding, m2, make_strategy, batch)
            full2 = b2.state_dict()["model"]
            cell = {
                "wrapper_kind_fsdp1": b1._wrapper_kind,
                "wrapper_kind_fsdp2": b2._wrapper_kind,
                "fsdp2_in_place": w2 is m2,
                "fsdp2_root_is_group": isinstance(w2, FSDPModule),
                "losses_fsdp1": l1,
                "losses_fsdp2": l2,
            }
            sharded = sharding != "NO_SHARD"
            if name == "mtp" and sharded:
                embed = w2.draft_model.embed_tokens.weight
                head_w = w2.draft_model.mtp.lm_head.weight
                cell["mtp_tied_same_object"] = embed is head_w
                cell["mtp_tied_is_dtensor"] = isinstance(embed, DTensor)
            if name == "peagle" and sharded:
                mask_hidden = w2.draft_model.mask_hidden
                cell["peagle_mask_hidden_is_dtensor"] = isinstance(mask_hidden, DTensor)
                cell["peagle_mask_hidden_local_rows"] = (
                    int(mask_hidden.to_local().shape[0])
                    if isinstance(mask_hidden, DTensor)
                    else None
                )
            if name == "dflash2":
                # DDP does not forward attribute reads; FSDP1's wrapper and the in-place FSDP2 module do
                inner1 = w1.module if b1._wrapper_kind == "ddp" else w1
                inner2 = w2.module if b2._wrapper_kind == "ddp" else w2
                cell["fused_head_fsdp2"] = bool(inner2._use_fused_unary_head(fused_probe))
                cell["fused_head_fsdp1"] = bool(inner1._use_fused_unary_head(fused_probe))
                cell["fsdp2_lm_head_plain"] = not isinstance(inner2.lm_head.weight, DTensor)
            if rank == 0:
                cell["weight_diff"] = _max_diff(full1, full2)
                cell["loss_rel"] = _rel(l2, l1)
                cell["gnorm_rel"] = _rel(g2, g1)
                cell["keys_equal"] = sorted(full1) == sorted(full2)
            else:
                cell["nonrank0_fsdp2_empty"] = full2 == {}
            results["matrix"][f"{name}/{sharding}"] = cell

    # ---- accumulation: toggle count + parameter residency (eagle3) -----------
    fresh_eagle3, eagle3_strategy, eagle3_batch = algorithms["eagle3"]
    for sharding in ("SHARD_GRAD_OP", "FULL_SHARD"):
        toggles = []
        sharded_after_micro = []
        state = {}

        def observe(wrapped, step, micro, stepped, state=state, out=sharded_after_micro):
            if "name" not in state:
                state["name"] = next(
                    n for n, p in wrapped.draft_model.named_parameters() if p.requires_grad
                )
                original = wrapped.set_requires_gradient_sync

                def counting(flag, *args, _original=original, **kw):
                    toggles.append(bool(flag))
                    return _original(flag, *args, **kw)

                wrapped.set_requires_gradient_sync = counting
            param = dict(wrapped.draft_model.named_parameters())[state["name"]]
            out.append(isinstance(param, DTensor))

        # the first train_step happens before observe can patch the toggle, so count from the second optimizer window onward
        train(FSDP2TrainingBackend, sharding, fresh_eagle3(), eagle3_strategy,
              eagle3_batch, observe=observe)
        results["accum"][sharding] = {
            "false_toggles": toggles.count(False),
            "true_toggles": toggles.count(True),
            "sharded_after_micro": sharded_after_micro,
        }

    # ---- collective schedule per micro-step: fsdp2 must match fsdp1 (eagle3) --
    import torch.distributed as dist

    comm = {}
    collectives = ("all_gather_single", "reduce_scatter_single", "all_reduce",
                   "all_gather_into_tensor", "reduce_scatter_tensor")

    def counting(name, real):
        def call(*args, **kwargs):
            comm[name] = comm.get(name, 0) + 1
            return real(*args, **kwargs)

        return call

    originals = {n: getattr(dist, n) for n in collectives if hasattr(dist, n)}
    for n, real in originals.items():
        setattr(dist, n, counting(n, real))
    try:
        for sharding in ("SHARD_GRAD_OP", "FULL_SHARD"):
            per_impl = {}
            for label, cls in (("fsdp1", FSDP1TrainingBackend), ("fsdp2", FSDP2TrainingBackend)):
                trace = []

                def observe(wrapped, step, micro, stepped, trace=trace):
                    torch.cuda.synchronize()
                    trace.append(dict(sorted(comm.items())))
                    comm.clear()

                train(cls, sharding, fresh_eagle3(), eagle3_strategy, eagle3_batch,
                      observe=observe, on_ready=lambda w, b: comm.clear())
                per_impl[label] = trace
            results["comm"][sharding] = per_impl
    finally:
        for n, real in originals.items():
            setattr(dist, n, real)

    # ---- NO_SHARD (DDP): no reduction before the boundary, both backends -----
    def grads_equal_across_ranks(param):
        g = param.grad.detach().float() if param.grad is not None else torch.zeros_like(param, dtype=torch.float32)
        gathered = [torch.empty_like(g) for _ in range(world_size)]
        dist.all_gather(gathered, g)
        return bool(torch.equal(gathered[0], gathered[1]))

    for label, cls in (("fsdp1", FSDP1TrainingBackend), ("fsdp2", FSDP2TrainingBackend)):
        equal = []

        def observe(wrapped, step, micro, stepped, out=equal):
            torch.cuda.synchronize()
            param = next(p for p in wrapped.module.draft_model.parameters() if p.requires_grad)
            out.append(grads_equal_across_ranks(param))

        train(cls, "NO_SHARD", fresh_eagle3(), eagle3_strategy, eagle3_batch, observe=observe)
        results["ddp_nosync"][label] = equal

    # ---- unequal valid-token counts per rank (loss_terms path) ---------------
    TokenSumStrategy = _toy_strategy_class()
    from specforge.runtime.contracts import TrainBatch

    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, 128, (world_size, 16), generator=g)
    labels = torch.randint(0, 128, (world_size, 16), generator=g)
    loss_mask = torch.zeros(world_size, 16, dtype=torch.long)
    loss_mask[0, :5] = 1  # rank 0: 5 valid tokens
    loss_mask[1, :11] = 1  # rank 1: 11 valid tokens
    local = TrainBatch(sample_ids=[f"r{rank}"], strategy="toy", tensors={
        "ids": ids[rank : rank + 1], "labels": labels[rank : rank + 1],
        "loss_mask": loss_mask[rank : rank + 1]})
    merged = TrainBatch(sample_ids=["r0", "r1"], strategy="toy", tensors={
        "ids": ids, "labels": labels, "loss_mask": loss_mask})

    def fresh_toy():
        torch.manual_seed(0)
        return _ToyComposite().cuda().to(torch.bfloat16)

    toy_gold = _cpu_sd(fresh_toy())

    def make_toy():
        model = fresh_toy()
        model.load_state_dict(toy_gold)
        return model

    def full_grads(wrapped, backend):
        # the gradients BF16Optimizer is about to clip, gathered to full shape
        kind = backend._wrapper_kind
        if kind == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            with FSDP.summon_full_params(wrapped, with_grads=True):
                return {
                    n.replace("_fsdp_wrapped_module.", ""): p.grad.detach().float().cpu().clone()
                    for n, p in wrapped.named_parameters() if p.grad is not None
                }
        out = {}
        for n, p in wrapped.named_parameters():
            if p.grad is not None:
                g = p.grad.full_tensor() if kind == "fsdp2" else p.grad
                out[n] = g.detach().float().cpu().clone()
        return out

    def grad_rel_diff(a, b):
        assert sorted(a) == sorted(b), set(a) ^ set(b)
        return max(
            float((a[k] - b[k]).abs().max() / b[k].abs().max().clamp_min(1e-8)) for k in b
        )

    # reference: one process on the merged batch; the denominator all-reduce of identical counts cancels
    ref_grads = []
    ref = make_toy()
    _, _, _, ref_gnorms = train(
        FSDP1TrainingBackend, "SHARD_GRAD_OP", ref, TokenSumStrategy, merged, wrap=False,
        steps=2, acc=1, before_step=lambda w, b: ref_grads.append(full_grads(w, b)),
    )
    ref_sd = _cpu_sd(ref)
    for label, backend_cls in (("fsdp1", FSDP1TrainingBackend), ("fsdp2", FSDP2TrainingBackend)):
        grads = []
        backend, _, losses, gnorms = train(
            backend_cls, "SHARD_GRAD_OP", make_toy(), TokenSumStrategy, local,
            steps=2, acc=1, before_step=lambda w, b: grads.append(full_grads(w, b)),
        )
        full = backend.state_dict()["model"]
        # mutation: without the global-denominator scaling the norm is off by world*acc/N
        _, _, _, mutant_gnorms = train(
            backend_cls, "SHARD_GRAD_OP", make_toy(), TokenSumStrategy, local,
            steps=1, acc=1, no_scale=True,
        )
        if rank == 0:
            results["tokens"][label] = {
                "weight_diff": _max_diff(full, ref_sd),
                "losses": losses,
                "gnorms": gnorms,
                "ref_gnorms": ref_gnorms,
                "gnorm_rel_diff": max(abs(g - r) / r for g, r in zip(gnorms, ref_gnorms)),
                "grad_rel_diff": max(grad_rel_diff(g, r) for g, r in zip(grads, ref_grads)),
                "mutant_gnorm_rel_diff": abs(mutant_gnorms[0] - ref_gnorms[0]) / ref_gnorms[0],
            }

    # ---- launch-level resume continuity, fsdp1 and fsdp2 ---------------------
    from specforge.algorithms.builtin import builtin_algorithm_registry
    from specforge.algorithms.common.providers import (
        MODEL_PROVENANCE_CONTRACT_KEY,
        OMITTED_STATE_FINGERPRINT_CONTRACT_KEY,
        StepRuntimeConfig,
        checkpoint_key_fingerprint,
    )
    from specforge.launch import build_offline_runtime
    from specforge.modeling.target.target_head import TargetHead
    from specforge.training.checkpoint import CheckpointManager

    algorithm = builtin_algorithm_registry().resolve("eagle3")
    head = TargetHead.from_pretrained(
        os.path.join(workdir, "target"), lm_head_key="lm_head.weight"
    )
    allowed_missing = {"embed_tokens.weight"}  # eagle3 omits its frozen embedding

    def runtime(model):
        return StepRuntimeConfig(
            options={},
            resume_contract={
                OMITTED_STATE_FINGERPRINT_CONTRACT_KEY: checkpoint_key_fingerprint(
                    model.draft_model, allowed_missing
                ),
                MODEL_PROVENANCE_CONTRACT_KEY: (("test_model_source", "fixture-v1"),),
            },
            allowed_missing_checkpoint_keys=allowed_missing,
        )

    def launch(model, *, out, max_steps, logged, resume_from=None):
        return build_offline_runtime(
            algorithm=algorithm,
            hidden_states_path=os.path.join(workdir, "features"),
            draft_model=model,
            target_head=head,
            optimizer_factory=_optimizer_factory,
            run_id="matrix-resume",
            output_dir=out,
            ttt_length=3,
            max_len=512,
            batch_size=1,
            accumulation_steps=2,
            num_epochs=3,
            max_steps=max_steps,
            # one schedule horizon for every run: the resume contract pins it
            total_steps=4,
            strategy_kwargs=runtime(model),
            resume_from=resume_from,
            logger=lambda metrics, step: logged.append((step, float(metrics["loss"]))),
            log_interval=1,
        )

    for version in (1, 2):
        with mock.patch.dict("os.environ", {"FSDP_VERSION": str(version)}):
            base = os.path.join(workdir, f"resume_v{version}")
            logged_a, logged_b, logged_c = [], [], []
            ta = launch(fresh_eagle3(), out=os.path.join(base, "a"), max_steps=4, logged=logged_a)
            ta.fit()  # the lifecycle writes the final checkpoint itself
            full_a = ta.backend.state_dict()["model"]
            tb = launch(fresh_eagle3(), out=os.path.join(base, "b"), max_steps=2, logged=logged_b)
            tb.fit()
            ckpt = CheckpointManager.resolve_resume_dir(os.path.join(base, "b"))
            tc = launch(fresh_eagle3(), out=os.path.join(base, "c"), max_steps=4,
                        logged=logged_c, resume_from=ckpt)
            tc.fit()
            full_c = tc.backend.state_dict()["model"]
            entry = {
                "backend": tc.backend.name,
                "global_step": tc.global_step,
                "losses_uninterrupted": logged_a,
                "losses_resumed": logged_b + logged_c,
            }
            if rank == 0:
                entry["weight_diff"] = _max_diff(full_a, full_c)
            results["launch_resume"][f"v{version}"] = entry

    with open(os.path.join(results_dir, f"rank{rank}.json"), "w") as f:
        json.dump(results, f, default=str)
    dist.barrier()
    dist.destroy_process_group()


@unittest.skipUnless(CUDA and NGPU >= WORLD, f"requires >={WORLD} GPUs")
class FSDP2MatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch.multiprocessing as mp

        from tests.test_runtime import _fixtures as fx

        cls.workdir = tempfile.mkdtemp(prefix="fsdp2_matrix_")
        fx.write_draft_config(os.path.join(cls.workdir, "draft.json"))
        fx.write_target_head_dir(os.path.join(cls.workdir, "target"))
        fx.write_vocab_mapping(os.path.join(cls.workdir, "vm.pt"))
        # 8 samples: 4 per rank = 2 optimizer steps per epoch at bs 1 x acc 2
        fx.write_offline_files(os.path.join(cls.workdir, "features"), n=8, seq=64)
        results_dir = os.path.join(cls.workdir, "results")
        os.makedirs(results_dir, exist_ok=True)
        mp.spawn(_matrix_worker, args=(WORLD, cls.workdir, results_dir), nprocs=WORLD, join=True)
        cls.results = []
        for rank in range(WORLD):
            with open(os.path.join(results_dir, f"rank{rank}.json")) as f:
                cls.results.append(json.load(f))

    def test_every_algorithm_matches_fsdp1_under_every_sharding_mode(self):
        r0, r1 = self.results
        for name in ALGORITHMS:
            for sharding in SHARDINGS:
                cell = r0["matrix"][f"{name}/{sharding}"]
                with self.subTest(algorithm=name, sharding=sharding):
                    self.assertTrue(cell["keys_equal"], cell)
                    if sharding == "NO_SHARD":
                        # DDP under both implementations: the same code path
                        self.assertEqual(cell["wrapper_kind_fsdp1"], "ddp")
                        self.assertEqual(cell["wrapper_kind_fsdp2"], "ddp")
                        self.assertEqual(cell["weight_diff"], 0.0, cell)
                        self.assertEqual(cell["losses_fsdp1"], cell["losses_fsdp2"])
                    else:
                        self.assertEqual(cell["wrapper_kind_fsdp2"], "fsdp2")
                        self.assertTrue(cell["fsdp2_in_place"])
                        self.assertTrue(cell["fsdp2_root_is_group"])
                        self.assertLess(cell["weight_diff"], 1e-2, cell)
                        self.assertLess(cell["loss_rel"], 2e-2, cell)
                        self.assertLess(cell["gnorm_rel"], 5e-2, cell)
                        self.assertTrue(
                            r1["matrix"][f"{name}/{sharding}"]["nonrank0_fsdp2_empty"]
                        )

    def test_mtp_tied_frozen_tables_stay_one_parameter(self):
        for sharding in ("SHARD_GRAD_OP", "FULL_SHARD"):
            cell = self.results[0]["matrix"][f"mtp/{sharding}"]
            self.assertTrue(cell["mtp_tied_same_object"], cell)
            self.assertTrue(cell["mtp_tied_is_dtensor"], cell)

    def test_peagle_single_row_parameter_shards_to_empty_rows(self):
        rows = [
            r["matrix"]["peagle/SHARD_GRAD_OP"]["peagle_mask_hidden_local_rows"]
            for r in self.results
        ]
        self.assertTrue(self.results[0]["matrix"]["peagle/SHARD_GRAD_OP"]["peagle_mask_hidden_is_dtensor"])
        self.assertEqual(rows, [1, 0], rows)

    def test_dflash2_fused_head_stays_eligible_under_fsdp2(self):
        for sharding in SHARDINGS:
            cell = self.results[0]["matrix"][f"dflash2/{sharding}"]
            with self.subTest(sharding=sharding):
                self.assertTrue(cell["fused_head_fsdp1"], cell)
                self.assertTrue(cell["fused_head_fsdp2"], cell)
                self.assertTrue(cell["fsdp2_lm_head_plain"], cell)

    def test_accumulation_toggle_count_and_parameter_residency(self):
        for rank_result in self.results:
            grad_op = rank_result["accum"]["SHARD_GRAD_OP"]
            full = rank_result["accum"]["FULL_SHARD"]
            # the toggle is patched after the first micro-step, so STEPS-1 windows are observed: each flips False once and True once
            self.assertEqual(grad_op["false_toggles"], (STEPS - 1) * (ACC - 1), grad_op)
            self.assertEqual(grad_op["true_toggles"], STEPS - 1, grad_op)
            self.assertEqual(full["false_toggles"], (STEPS - 1) * (ACC - 1), full)
            # SHARD_GRAD_OP keeps the unsharded parameters resident until the boundary; FULL_SHARD reshards after every micro-step
            self.assertEqual(grad_op["sharded_after_micro"], [False, True] * STEPS, grad_op)
            self.assertEqual(full["sharded_after_micro"], [True, True] * STEPS, full)

    def test_collective_schedule_matches_fsdp1_per_micro_step(self):
        # collective counts per micro-step must equal FSDP1's in both modes (root keeps params for backward)
        for rank_result in self.results:
            for sharding, per_impl in rank_result["comm"].items():
                self.assertEqual(per_impl["fsdp2"], per_impl["fsdp1"], (sharding, per_impl))

    def test_no_shard_does_not_reduce_before_the_boundary(self):
        # DDP decides at forward time; distinct data per rank, so local grads differ until the boundary reduces
        for rank_result in self.results:
            for label, equal in rank_result["ddp_nosync"].items():
                self.assertEqual(equal, [False, True] * STEPS, (label, equal))

    def test_unequal_token_counts_match_merged_batch_reference(self):
        # compare what the optimizer clips: AdamW + clipping would hide a wrong denominator in the weights alone
        tokens = self.results[0]["tokens"]
        for label in ("fsdp1", "fsdp2"):
            cell = tokens[label]
            self.assertLess(cell["gnorm_rel_diff"], 2e-2, cell)
            self.assertLess(cell["grad_rel_diff"], 3e-2, cell)
            self.assertLess(cell["weight_diff"], 1e-2, cell)
            # sensitivity: a no-op scale_gradients is off by world*acc/N = 8x here
            self.assertGreater(cell["mutant_gnorm_rel_diff"], 1.0, cell)

    def test_launch_level_resume_matches_uninterrupted_run(self):
        for version, name in ((1, "fsdp1"), (2, "fsdp2")):
            entry = self.results[0]["launch_resume"][f"v{version}"]
            with self.subTest(fsdp_version=version):
                self.assertEqual(entry["backend"], name)
                self.assertEqual(entry["global_step"], 4)
                self.assertEqual(entry["losses_resumed"], entry["losses_uninterrupted"], entry)
                self.assertEqual(entry["weight_diff"], 0.0, entry)


if __name__ == "__main__":
    unittest.main(verbosity=2)
