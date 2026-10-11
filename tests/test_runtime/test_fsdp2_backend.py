# coding=utf-8
"""FSDP2 backend (``training.fsdp_version: 2``): wrap layout and parity.

1 GPU: fully_shard layout for a root-only draft (EAGLE3), a per-block draft
with replicated frozen tables (DFlash) and nested block matches (children first).
2 GPUs: fsdp1 vs fsdp2 on DISTINCT per-rank micro-batches (EAGLE3; the other
algorithms and sharding modes are in ``test_fsdp2_matrix.py``),
the gradient-sync toggle vs reducing every micro-step (bitwise), and a resume
through today's rank-local checkpoint format (bitwise), and an export round
trip (every rank saves, rank0 exports to HF / sglang and reloads bitwise).  Mirrors
``test_no_sync_equiv.py``; run on the H200 box via rcli.
"""

import json
import os
import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()
NGPU = torch.cuda.device_count() if CUDA else 0
WORLD = 2
BS = 2
ACC = 2
STEPS = 4


def _optimizer_factory(module):
    from specforge.optimizer import BF16Optimizer

    return BF16Optimizer(
        module, lr=1e-3, max_grad_norm=0.5, warmup_ratio=0.0, total_steps=100
    )


@unittest.skipUnless(CUDA, "fully_shard layout tests need CUDA")
class FSDP2LayoutTest(unittest.TestCase):
    def test_eagle3_root_only_and_dflash_per_block(self):
        from torch.distributed.fsdp import FSDPModule
        from torch.distributed.tensor import DTensor

        from specforge.training.backend import FSDP2TrainingBackend, ParallelConfig
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29592")
        workdir = tempfile.mkdtemp(prefix="fsdp2_layout_")
        pc = ParallelConfig.from_distributed()
        pc.fsdp_version = 2

        eagle3_model, _ = fx.build_eagle3(workdir)
        keys_before = set(eagle3_model.state_dict())
        backend = FSDP2TrainingBackend(pc, optimizer_factory=_optimizer_factory)
        wrapped = backend.prepare_model(
            eagle3_model, optimizer_target=eagle3_model.draft_model
        )
        # in place: the composite is the root group, the draft is not a group
        self.assertIs(wrapped, eagle3_model)
        self.assertIsInstance(wrapped, FSDPModule)
        self.assertNotIsInstance(wrapped.draft_model, FSDPModule)
        self.assertEqual(backend.auto_wrap_block_classes, set())
        self.assertTrue(all(isinstance(p, DTensor) for p in wrapped.parameters()))
        # the optimizer works on plain local shards
        self.assertTrue(
            all(
                not isinstance(mp, DTensor) for mp in backend.optimizer.fp32_params
            )
        )
        full = backend.state_dict()["model"]
        self.assertEqual(set(full), keys_before)
        self.assertTrue(all(v.device.type == "cpu" for v in full.values()))

        dflash_model, _, _, _ = fx.build_dflash(
            workdir, hidden=64, block_size=4, num_anchors=8, attention_backend="sdpa"
        )
        backend2 = FSDP2TrainingBackend(pc, optimizer_factory=_optimizer_factory)
        wrapped2 = backend2.prepare_model(
            dflash_model, optimizer_target=dflash_model.draft_model
        )
        self.assertEqual(
            {c.__name__ for c in backend2.auto_wrap_block_classes},
            {"Qwen3DFlashDecoderLayer"},
        )
        self.assertTrue(
            all(isinstance(layer, FSDPModule) for layer in wrapped2.draft_model.layers)
        )
        # frozen target tables stay replicated plain tensors (fused head reads them)
        self.assertNotIsInstance(wrapped2.lm_head.weight, DTensor)
        self.assertNotIsInstance(wrapped2.embed_tokens.weight, DTensor)
        self.assertTrue(
            all(
                b.dtype == torch.float32
                for b in wrapped2.buffers()
                if b.is_floating_point()
            )
        )

    def test_nested_block_matches_wrap_children_first(self):
        # nested _no_split_modules matches: inner groups first, or the outer group owns the inner params too
        import torch.nn as nn
        from torch.distributed.tensor import DTensor

        from specforge.training.backend import FSDP2TrainingBackend, ParallelConfig

        class _Inner(nn.Module):
            def __init__(self, h):
                super().__init__()
                self.fc = nn.Linear(h, h, bias=False)

            def forward(self, x):
                return torch.nn.functional.silu(self.fc(x)) + x

        class _Outer(nn.Module):
            def __init__(self, h):
                super().__init__()
                self.pre = nn.Linear(h, h, bias=False)
                self.inners = nn.ModuleList([_Inner(h), _Inner(h)])

            def forward(self, x):
                x = self.pre(x)
                for inner in self.inners:
                    x = inner(x)
                return x

        class _Draft(nn.Module):
            _no_split_modules = ["_Outer", "_Inner"]

            def __init__(self, h=32):
                super().__init__()
                self.layers = nn.ModuleList([_Outer(h), _Outer(h)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        class _Composite(nn.Module):
            def __init__(self, h=32):
                super().__init__()
                self.draft_model = _Draft(h)
                self.head = nn.Linear(h, 1, bias=False)

            def forward(self, x):
                return self.head(self.draft_model(x)).float().pow(2).mean()

        torch.manual_seed(0)
        model = _Composite().cuda().to(torch.bfloat16)
        pc = ParallelConfig.from_distributed()
        pc.fsdp_version = 2
        pc.sharding_strategy = "FULL_SHARD"
        backend = FSDP2TrainingBackend(pc)
        wrapped = backend.prepare_model(model, optimizer_target=model.draft_model)
        self.assertTrue(all(isinstance(p, DTensor) for p in wrapped.parameters()))
        self.assertEqual(backend.auto_wrap_block_classes, {_Outer, _Inner})
        loss = wrapped(torch.randn(4, 32, device="cuda", dtype=torch.bfloat16))
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in wrapped.parameters()))


def _parity_worker(rank, world_size, workdir, results_dir):
    import torch.distributed as dist

    from tests.test_runtime import _fixtures as fx

    fx.init_rank_distributed(rank, world_size, port="29593")
    torch.use_deterministic_algorithms(True, warn_only=True)

    from specforge.algorithms.eagle3.model import OnlineEagle3Model
    from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
    from specforge.modeling.target.target_head import TargetHead
    from specforge.runtime.contracts import TrainBatch
    from specforge.training.backend import (
        FSDP1TrainingBackend,
        FSDP2TrainingBackend,
        ParallelConfig,
    )
    from specforge.training.controller import TrainerCore
    from specforge.training.strategies.base import (
        Eagle3TrainStrategy,
        StepContext,
    )

    def cpu_sd(module):
        return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

    def max_diff(a, b):
        assert sorted(a) == sorted(b), set(a) ^ set(b)
        worst = 0.0
        for key in a:
            x, y = a[key].cpu(), b[key].cpu()
            if x.is_floating_point():
                worst = max(worst, float((x.float() - y.float()).abs().max()))
            elif not torch.equal(x, y):
                worst = float("inf")
        return worst

    def train(backend_cls, model, make_strategy, batch, *, steps=STEPS,
              per_micro=False, sharding="SHARD_GRAD_OP"):
        pc = ParallelConfig.from_distributed()
        pc.sharding_strategy = sharding
        pc.fsdp_version = 2 if backend_cls is FSDP2TrainingBackend else 1
        backend = backend_cls(pc, optimizer_factory=_optimizer_factory)
        wrapped = backend.prepare_model(model, optimizer_target=model.draft_model)
        strategy = make_strategy(wrapped)
        losses = []
        if per_micro:
            for step in range(steps):
                ctx = StepContext(global_step=step, total_steps=steps)
                for _ in range(ACC):
                    out = strategy.forward_loss(batch, ctx)
                    backend.backward(out.loss / ACC, is_boundary=True)
                backend.step()
                losses.append(float(out.loss.detach()))
            return backend, losses
        core = TrainerCore(strategy, backend, accumulation_steps=ACC)
        for step in range(steps):
            ctx = StepContext(global_step=step, total_steps=steps)
            for _ in range(ACC):
                result = core.train_step(batch, ctx)
            losses.append(float(result.loss))
        return backend, losses

    results = {}

    # ---- EAGLE3: root-only groups --------------------------------------------
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

    loader = fx.build_offline_eagle3_loader(
        os.path.join(workdir, "features"),
        batch_size=BS,
        run_id="fsdp2-parity",
        ttt_length=3,
        max_len=512,
        ref_slice=slice(rank * BS, (rank + 1) * BS),  # DISTINCT shard per rank
    )
    batch = next(iter(loader))
    strategy = lambda w: Eagle3TrainStrategy(w, target_head=head)  # noqa: E731

    for sharding in ("SHARD_GRAD_OP", "FULL_SHARD"):
        b1, l1 = train(FSDP1TrainingBackend, fresh_eagle3(), strategy, batch, sharding=sharding)
        full1 = b1.state_dict()["model"]
        b2, l2 = train(FSDP2TrainingBackend, fresh_eagle3(), strategy, batch, sharding=sharding)
        full2 = b2.state_dict()["model"]
        b3, _ = train(FSDP2TrainingBackend, fresh_eagle3(), strategy, batch,
                      per_micro=True, sharding=sharding)
        full3 = b3.state_dict()["model"]
        if rank == 0:
            results[f"eagle3_{sharding}_parity_weight_diff"] = max_diff(full1, full2)
            results[f"eagle3_{sharding}_parity_loss_rel"] = max(
                abs(x - y) / max(abs(y), 1e-8) for x, y in zip(l2, l1)
            )
            results[f"eagle3_{sharding}_toggle_equiv_weight_diff"] = max_diff(full2, full3)
            results[f"eagle3_{sharding}_state_dict_keys_equal"] = sorted(full1) == sorted(full2)
        else:
            results[f"eagle3_{sharding}_nonrank0_state_empty"] = full2 == {}
        if sharding == "SHARD_GRAD_OP":
            full2_default, l2_default = full2, l2
    full2, l2 = full2_default, l2_default

    # resume through the rank-local format: 2 steps, save, fresh model, 2 more
    bs1, ls1 = train(FSDP2TrainingBackend, fresh_eagle3(), strategy, batch, steps=2)
    state = bs1.state_dict()
    model_path = os.path.join(workdir, "resume_model.pt")
    if rank == 0:
        torch.save(state["model"], model_path)
    dist.barrier()
    resumed = fresh_eagle3()
    resumed.load_state_dict(torch.load(model_path, map_location="cpu"))  # before wrap
    pc = ParallelConfig.from_distributed()
    pc.fsdp_version = 2
    bs2 = FSDP2TrainingBackend(pc, optimizer_factory=_optimizer_factory)
    ws2 = bs2.prepare_model(resumed, optimizer_target=resumed.draft_model)
    bs2.load_state_dict({"model": None, "optimizer": state["optimizer"], "rng": state["rng"]})
    core = TrainerCore(strategy(ws2), bs2, accumulation_steps=ACC)
    ls2 = []
    for step in range(2, STEPS):
        ctx = StepContext(global_step=step, total_steps=STEPS)
        for _ in range(ACC):
            result = core.train_step(batch, ctx)
        ls2.append(float(result.loss))
    full_resumed = bs2.state_dict()["model"]
    if rank == 0:
        results["eagle3_resume_weight_diff"] = max_diff(full_resumed, full2)
        results["eagle3_resume_losses_equal"] = ls1 + ls2 == l2

    # ---- export round trip: every rank saves, rank0 exports and reloads ------
    from safetensors.torch import load_file, save_file

    from specforge.export import export_to_hf, export_to_sglang
    from specforge.training.checkpoint import CheckpointManager

    full = bs2.state_dict()  # collective gather, {} off rank0
    cfg_path = os.path.join(workdir, "draft.json")
    mgr = CheckpointManager(os.path.join(workdir, "export_run"), "fsdp2-export")
    shared = None
    if rank == 0:
        filtered = strategy(ws2).checkpoint_state_filter(full["model"])
        shared = {
            "draft_state_dict": filtered,
            "global_step": STEPS,
            "epoch": 0,
            "epoch_batch": 0,
            "epoch_samples": 0,
            "strategy": "eagle3",
            "run_id": "fsdp2-export",
            "world_size": world_size,
            "training_backend": "fsdp2",
        }
    ckpt_dir = mgr.save(  # collective: barriers inside, so every rank calls it
        shared, STEPS, rank_state={"optimizer": full["optimizer"], "rng": full["rng"]}
    )
    if rank == 0:
        emb_dir = os.path.join(workdir, "emb_src")
        os.makedirs(emb_dir, exist_ok=True)
        embed = full["model"]["draft_model.embed_tokens.weight"].contiguous()
        save_file({"model.embed_tokens.weight": embed}, os.path.join(emb_dir, "model.safetensors"))
        with open(os.path.join(emb_dir, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {}, "weight_map": {"model.embed_tokens.weight": "model.safetensors"}}, f)
        hf_dir = export_to_hf(
            ckpt_dir, cfg_path, os.path.join(workdir, "hf_export"), embedding_source=emb_dir
        )
        reloaded = AutoDraftModel.from_pretrained(hf_dir, torch_dtype=torch.bfloat16).state_dict()
        results["fsdp2_export_hf_mismatch"] = [
            k for k, v in filtered.items() if not torch.equal(v.cpu(), reloaded[k].cpu())
        ]
        sg_dir = export_to_sglang(
            ckpt_dir, cfg_path, os.path.join(workdir, "sglang_export"),
            vocab_mapping_path=os.path.join(workdir, "vm.pt"),
        )
        served = load_file(os.path.join(sg_dir, "model.safetensors"))
        results["fsdp2_export_sglang_mismatch"] = [
            k for k, v in filtered.items()
            if k not in served or not torch.equal(v.cpu(), served[k].cpu())
        ]

    with open(os.path.join(results_dir, f"rank{rank}.json"), "w") as f:
        json.dump(results, f)
    dist.barrier()
    dist.destroy_process_group()


@unittest.skipUnless(CUDA and NGPU >= WORLD, f"requires >={WORLD} GPUs")
class FSDP2ParityTest(unittest.TestCase):
    def test_fsdp2_matches_fsdp1_and_resumes_bitwise(self):
        import torch.multiprocessing as mp

        from tests.test_runtime import _fixtures as fx

        workdir = tempfile.mkdtemp(prefix="fsdp2_parity_")
        fx.write_draft_config(os.path.join(workdir, "draft.json"))
        fx.write_target_head_dir(os.path.join(workdir, "target"))
        fx.write_vocab_mapping(os.path.join(workdir, "vm.pt"))
        fx.write_offline_files(os.path.join(workdir, "features"), n=WORLD * BS, seq=64)
        results_dir = os.path.join(workdir, "results")
        os.makedirs(results_dir, exist_ok=True)
        mp.spawn(
            _parity_worker, args=(WORLD, workdir, results_dir), nprocs=WORLD, join=True
        )

        with open(os.path.join(results_dir, "rank0.json")) as f:
            r0 = json.load(f)
        with open(os.path.join(results_dir, "rank1.json")) as f:
            r1 = json.load(f)
        # fsdp1 vs fsdp2 differ by bf16 reduce-scatter rounding (AVG vs pre-divide): tolerance, not equality
        for sharding in ("SHARD_GRAD_OP", "FULL_SHARD"):
            with self.subTest(sharding=sharding):
                self.assertLess(r0[f"eagle3_{sharding}_parity_weight_diff"], 1e-2, r0)
                self.assertLess(r0[f"eagle3_{sharding}_parity_loss_rel"], 2e-2, r0)
                self.assertTrue(r0[f"eagle3_{sharding}_state_dict_keys_equal"], r0)
                # within fsdp2 the accumulation toggle is exact
                self.assertEqual(r0[f"eagle3_{sharding}_toggle_equiv_weight_diff"], 0.0, r0)
                self.assertTrue(r1[f"eagle3_{sharding}_nonrank0_state_empty"], r1)
        # the resume through today's rank-local format is exact
        self.assertEqual(r0["eagle3_resume_weight_diff"], 0.0, r0)
        self.assertTrue(r0["eagle3_resume_losses_equal"], r0)
        # fsdp2 checkpoint -> HF / sglang export -> reload, bitwise
        self.assertEqual(r0["fsdp2_export_hf_mismatch"], [], r0)
        self.assertEqual(r0["fsdp2_export_sglang_mismatch"], [], r0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
