# coding=utf-8
"""trim_backbone_rows: TTT steps >= 1 forward only rows that can still emit loss.

Layers (CI keeps guarding the union-row math even without GPUs):

1. ``TestBackboneRowsGolden`` (CPU) -- hand-derived literal tables for
   ``R_i = union_{j >= i} { s - j : s in sup, 0 <= s - j < chunk_len }``:
   the nested-union semantics, the overlap-tail bound, dead-tail padding and
   non-USP back-compat. (The naive "reuse ``sup`` at every step" reading is
   wrong -- loss rows shift left once per TTT step -- and these tables pin the
   corrected math.)
2. ``TestEquivTrimBackboneSdpa`` (one GPU) -- per-step loss parity, backbone
   trim ON vs OFF (both with trim_loss_positions on): an all-supervised mask
   where R_i covers every row (trivial-equality boundary: any row/position/
   mask error is exposed without tolerance cover), plus prompt-heavy,
   block-boundary, and isolated-point masks, plus a total-grad-norm check.
3. ``TestEquivTrimBackboneFa`` -- same masks with attention_backend=fa
   (step 0 native flash, trimmed steps on the shared manual path).
4. ``TestEquivTrimBackboneUspFourRank`` -- ring-4 offline pipeline: the
   trimmed steps bypass ring/Ulysses and attend an all-gathered step-0 K/V,
   so per-rank row counts may differ freely; masks include supervision
   straddling every rank boundary and a tail-only rank (dead steps + ranks
   falling back to the untrimmed path).
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import torch

CUDA = torch.cuda.is_available()
NGPU = torch.cuda.device_count() if CUDA else 0
WORLD_SIZE = 4
SEQ = 48
TTT = 3


def _has_standard_flash_attention() -> bool:
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
        from flash_attn.bert_padding import pad_input, unpad_input  # noqa: F401
        from flash_attn.flash_attn_interface import (  # noqa: F401
            _flash_attn_varlen_backward,
        )
    except Exception:
        return False
    return True


class TestBackboneRowsGolden(unittest.TestCase):
    """Hand-derived expected outputs for the B-level row sets (CPU only)."""

    def _mk(self, mask_list, seed=1, vocab=32, draft=8):
        g = torch.Generator().manual_seed(seed)
        ids = torch.randperm(vocab, generator=g)[:draft].sort().values
        t2d = torch.zeros(vocab, dtype=torch.bool)
        t2d[ids] = True
        L = len(mask_list)
        lm = torch.tensor(mask_list, dtype=torch.long).view(1, L, 1)
        tgt = torch.randn(1, L, vocab, generator=g)
        return tgt, t2d, lm

    def test_union_with_tail_bound(self):
        # C=6, k=3, mask=[0,0,0,0,1,1,1,0] -> sup={4,5,6}
        #   loss rows: j0={4,5} j1={3,4,5} j2={2,3,4}
        #   R_1 = j1 U j2 = {2,3,4,5};  R_2 = j2 = {2,3,4}
        from specforge.algorithms.eagle3.model import _build_trim_pack

        tgt, t2d, lm = self._mk([0, 0, 0, 0, 1, 1, 1, 0])
        p = _build_trim_pack(tgt, t2d, lm, length=3, chunk_len=6, backbone_rows=True)
        self.assertEqual(p["b_rows_steps"][1].tolist(), [2, 3, 4, 5])
        self.assertEqual(p["b_rows_steps"][2].tolist(), [2, 3, 4])
        # step-1 hidden comes from the full-length step-0 output
        # (R_i nested in R_{i-1}: step-i prev-selection = b_diag_sels[i][i-1])
        # loss rows inside the compact row space
        self.assertEqual(p["b_loss_sel"][1].tolist(), [1, 2, 3])  # {3,4,5} in R_1
        self.assertEqual(p["b_loss_sel"][2].tolist(), [0, 1, 2])  # {2,3,4} in R_2
        # diagonal map: R_2 inside R_1
        self.assertEqual(p["b_diag_sels"][2][1].tolist(), [0, 1, 2])

    def test_allones_covers_every_row(self):
        # mask = all ones over 8 slots, C=6, k=3 -> R_1 = R_2 = {0..5}
        from specforge.algorithms.eagle3.model import _build_trim_pack

        tgt, t2d, lm = self._mk([1] * 8)
        p = _build_trim_pack(tgt, t2d, lm, length=3, chunk_len=6, backbone_rows=True)
        self.assertEqual(p["b_rows_steps"][1].tolist(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(p["b_rows_steps"][2].tolist(), [0, 1, 2, 3, 4, 5])

    def test_isolated_points_shrinking_chain(self):
        # non-USP: C=L=8, k=3, mask=[1,0,0,1,0,0,0,0] -> sup={0,3}
        #   loss rows: j0={0,3} j1={2} j2={1}
        #   R_1 = {1,2};  R_2 = {1}
        from specforge.algorithms.eagle3.model import _build_trim_pack

        tgt, t2d, lm = self._mk([1, 0, 0, 1, 0, 0, 0, 0])
        p = _build_trim_pack(tgt, t2d, lm, length=3, backbone_rows=True)
        self.assertEqual(p["b_rows_steps"][1].tolist(), [1, 2])
        self.assertEqual(p["b_rows_steps"][2].tolist(), [1])
        self.assertEqual(p["b_diag_sels"][2][1].tolist(), [0])

    def test_dead_tail_keeps_nested_padding(self):
        # sup so late that the last step has no reachable row: the padded row
        # must come from the previous set (nested chain survives).
        from specforge.algorithms.eagle3.model import _build_trim_pack

        tgt, t2d, lm = self._mk([0, 1, 0, 0, 0, 0, 0, 0])
        # sup={1}: loss rows j0={1} j1={0} j2={} (1-2 < 0)
        p = _build_trim_pack(tgt, t2d, lm, length=3, backbone_rows=True)
        self.assertEqual(p["b_rows_steps"][1].tolist(), [0])
        self.assertEqual(p["b_rows_steps"][2].tolist(), [0])  # padded from R_1
        self.assertEqual(p["nrows_steps"][2], 0)

    def test_off_flag_adds_no_keys(self):
        from specforge.algorithms.eagle3.model import _build_trim_pack

        tgt, t2d, lm = self._mk([0, 0, 0, 0, 1, 1, 1, 0])
        p = _build_trim_pack(tgt, t2d, lm, length=3, chunk_len=6)
        self.assertNotIn("b_rows_steps", p)


def _masks_single():
    m = {}
    allones = torch.ones(SEQ, dtype=torch.long)
    m["allones"] = allones
    prompt_heavy = torch.zeros(SEQ, dtype=torch.long)
    prompt_heavy[SEQ // 2 :] = 1
    m["prompt_heavy"] = prompt_heavy
    blocks = torch.zeros(SEQ, dtype=torch.long)
    blocks[[10, 11, 12, 13, 22, 23, 24, 25, 34, 35, 36, 37]] = 1
    m["blocks"] = blocks
    isolated = torch.zeros(SEQ, dtype=torch.long)
    isolated[[0, 9, 21, 33, 45]] = 1
    m["isolated"] = isolated
    return m


class _SingleRankEquivBase(unittest.TestCase):
    backend = "sdpa"
    allones_tol = 1e-6

    @classmethod
    def _build(cls, workdir):
        from specforge.algorithms.eagle3.model import OnlineEagle3Model
        from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
        from specforge.modeling.target.target_head import TargetHead
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29881")
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        fx.write_draft_config(os.path.join(workdir, "draft.json"))
        fx.write_target_head_dir(os.path.join(workdir, "target"))
        fx.write_vocab_mapping(os.path.join(workdir, "vm.pt"))
        cfg = AutoDraftModelConfig.from_file(os.path.join(workdir, "draft.json"))
        dm = AutoDraftModel.from_config(
            cfg, attention_backend=cls.backend, torch_dtype=torch.bfloat16
        ).cuda()
        dm.load_vocab_mapping(os.path.join(workdir, "vm.pt"))
        dm.freeze_embedding()
        model = OnlineEagle3Model(
            draft_model=dm, length=TTT, attention_backend=cls.backend
        ).cuda()
        head = TargetHead.from_pretrained(
            os.path.join(workdir, "target"), lm_head_key="lm_head.weight"
        )
        return model, head

    def _batch(self, head, loss_mask):
        from tests.test_runtime import _fixtures as fx

        g = torch.Generator().manual_seed(11)
        input_ids = torch.randint(0, fx.V, (1, SEQ), generator=g)
        hidden = torch.randn(1, SEQ, fx.H, generator=g).to(torch.bfloat16)
        aux = torch.randn(1, SEQ, 3 * fx.H, generator=g).to(torch.bfloat16)
        input_ids, target_hidden, lm = head.preprocess(
            input_ids, hidden, loss_mask.view(1, SEQ)
        )
        target = head(target_hidden.cuda())
        return dict(
            input_ids=input_ids.cuda(),
            attention_mask=torch.ones(1, SEQ, device="cuda"),
            loss_mask=lm.cuda(),
            target=target,
            hidden_states=aux.cuda(),
        )

    def _step_losses(self, model, batch, trim_backbone):
        with torch.no_grad():
            plosses, *_ = model(
                trim_loss_positions=True,
                trim_backbone_rows=trim_backbone,
                **batch,
            )
        return [float(p.item()) for p in plosses]

    def _grad_norm(self, model, batch, trim_backbone):
        model.zero_grad(set_to_none=True)
        plosses, *_ = model(
            trim_loss_positions=True, trim_backbone_rows=trim_backbone, **batch
        )
        loss = sum(0.8**i * plosses[i] for i in range(len(plosses)))
        loss.backward()
        total = torch.zeros((), device="cuda", dtype=torch.float64)
        for p in model.parameters():
            if p.grad is not None:
                total += (p.grad.double() ** 2).sum()
        model.zero_grad(set_to_none=True)
        return float(total.sqrt().item())

    def _run_case(self, name, mask, tol_fn):
        workdir = tempfile.mkdtemp(prefix="trim_b_")
        self.addCleanup(shutil.rmtree, workdir, ignore_errors=True)
        torch.use_deterministic_algorithms(True, warn_only=True)
        model, head = self._build(workdir)
        model.train()
        batch = self._batch(head, mask)
        off = self._step_losses(model, batch, False)
        on = self._step_losses(model, batch, True)
        self.assertEqual(len(off), TTT)
        for j, (a, b) in enumerate(zip(off, on)):
            self.assertLessEqual(
                abs(a - b), tol_fn(a), msg=f"{name} step{j}: off={a} on={b}"
            )
        g_off = self._grad_norm(model, batch, False)
        g_on = self._grad_norm(model, batch, True)
        self.assertLessEqual(
            abs(g_on - g_off), 5e-3 * g_off, msg=f"{name} grads: {g_off} vs {g_on}"
        )

    def test_allones_exact(self):
        self._run_case(
            "allones", _masks_single()["allones"], lambda a: self.allones_tol
        )

    def test_sdpa_core_forced(self):
        # Force the long-k0 SDPA core (normally k0_len >= 8192) through the
        # same all-ones + adversarial comparisons; measured residual is ~1e-5
        # (fused-kernel vs native op ordering), far below any discrete error.
        from specforge.modeling.draft import llama3_eagle as le

        with mock.patch.object(le, "_TRIM_SDPA_MIN_K0", 1):
            masks = _masks_single()
            for name in ("allones", "isolated"):
                with self.subTest(mask=name):
                    self._run_case(name, masks[name], lambda a: 5e-4)

    def test_adversarial_masks(self):
        masks = _masks_single()
        for name in ("prompt_heavy", "blocks", "isolated"):
            with self.subTest(mask=name):
                self._run_case(name, masks[name], lambda a: max(1e-3 * abs(a), 1e-4))


@unittest.skipUnless(CUDA, "requires one CUDA device")
class TestEquivTrimBackboneSdpa(_SingleRankEquivBase):
    backend = "sdpa"
    allones_tol = 1e-6


@unittest.skipUnless(
    CUDA and _has_standard_flash_attention(),
    "requires CUDA and flash-attn",
)
class TestEquivTrimBackboneFa(_SingleRankEquivBase):
    backend = "fa"
    # step 0 stays on flash while the reference path keeps flash at every
    # step; the trimmed steps change implementation, so bit-level equality is
    # not expected even for the all-ones mask.
    allones_tol = 1e-5


def _write_usp_workdir(workdir):
    from tests.test_runtime import _fixtures as fx

    fx.write_draft_config(os.path.join(workdir, "draft.json"))
    fx.write_target_head_dir(os.path.join(workdir, "target"))
    fx.write_vocab_mapping(os.path.join(workdir, "vocab_mapping.pt"))
    masks = {}
    m1 = torch.ones(SEQ, dtype=torch.long)
    m1[-1] = 0
    masks["allones"] = m1
    m3 = torch.zeros(SEQ, dtype=torch.long)
    m3[[10, 11, 12, 13, 22, 23, 24, 25, 34, 35, 36, 37]] = 1
    masks["boundary"] = m3
    m4 = torch.zeros(SEQ, dtype=torch.long)
    m4[[12, 13]] = 1
    masks["tailonly"] = m4
    g = torch.Generator().manual_seed(11)
    base_input = torch.randint(0, fx.V, (SEQ,), generator=g)
    base_hid = torch.randn(1, SEQ, fx.H, generator=g).to(torch.bfloat16)
    base_aux = torch.randn(1, SEQ, 3 * fx.H, generator=g).to(torch.bfloat16)
    for name, lm in masks.items():
        d = os.path.join(workdir, f"features_{name}")
        os.makedirs(d, exist_ok=True)
        torch.save(
            {
                "input_ids": base_input.clone(),
                "loss_mask": lm.clone(),
                "hidden_state": base_hid.clone(),
                "aux_hidden_state": base_aux.clone(),
            },
            os.path.join(d, "0000.ckpt"),
        )


def _usp_worker(rank, world_size, port, workdir):
    from tests.test_runtime import _fixtures as fx

    fx.init_rank_distributed(
        rank, world_size, tp_size=1, sp_ulysses_size=1, sp_ring_size=4, port=str(port)
    )
    try:
        import torch.distributed as dist

        from specforge.algorithms.builtin import builtin_algorithm_registry
        from specforge.algorithms.eagle3.model import OnlineEagle3Model
        from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
        from specforge.modeling.target.target_head import TargetHead
        from specforge.runtime.data_plane import FeatureDataLoader, LocalFeatureStore

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.use_deterministic_algorithms(True, warn_only=True)
        cfg = AutoDraftModelConfig.from_file(os.path.join(workdir, "draft.json"))
        dm = AutoDraftModel.from_config(
            cfg, attention_backend="usp", torch_dtype=torch.bfloat16
        ).cuda()
        dm.load_vocab_mapping(os.path.join(workdir, "vocab_mapping.pt"))
        dm.freeze_embedding()
        model = OnlineEagle3Model(
            draft_model=dm, length=TTT, attention_backend="usp"
        ).cuda()
        model.train()
        target_head = TargetHead.from_pretrained(
            os.path.join(workdir, "target"), lm_head_key="lm_head.weight"
        )
        algorithm = builtin_algorithm_registry().resolve("eagle3")
        provider = algorithm.providers.offline_for("text")

        results = {}
        for case in ("allones", "boundary", "tailonly"):
            refs = provider.build_reader(
                os.path.join(workdir, f"features_{case}"),
                run_id=f"trimb-{case}",
                ttt_length=TTT,
                max_len=SEQ,
            ).read()
            loader = FeatureDataLoader(
                LocalFeatureStore(f"trimb-{case}-{rank}"),
                refs=refs,
                batch_size=1,
                collate_fn=provider.build_collator(),
                per_sample_transform=provider.build_normalizer(
                    SEQ, ttt_length=TTT, use_usp_preprocess=True
                ),
                strategy=algorithm.name,
            )
            batch = next(iter(loader))

            def step_losses(trim_backbone):
                strat = algorithm.providers.step.build(
                    model,
                    target_head=target_head,
                    trim_loss_positions=True,
                    trim_backbone_rows=trim_backbone,
                )
                with torch.no_grad():
                    out = strat.forward_loss(batch)
                return [float(p.item()) for p in out.metrics["plosses"]]

            results[case] = {"off": step_losses(False), "on": step_losses(True)}

        gathered = [None] * world_size
        dist.all_gather_object(gathered, results)
        if rank == 0:
            with open(os.path.join(workdir, "results.json"), "w") as fh:
                json.dump(gathered, fh)
        dist.barrier()
    finally:
        from specforge.distributed import destroy_distributed

        destroy_distributed()


def _usp_worker_forced_sdpa(rank, world_size, port, workdir):
    from specforge.modeling.draft import llama3_eagle as le

    le._TRIM_SDPA_MIN_K0 = 1  # force the long-k0 SDPA core in every worker
    _usp_worker(rank, world_size, port, workdir)


@unittest.skipUnless(
    CUDA and NGPU >= WORLD_SIZE and _has_standard_flash_attention(),
    "requires four CUDA devices and the standard flash-attn USP interfaces",
)
class TestEquivTrimBackboneUspFourRank(unittest.TestCase):
    def test_sdpa_core_matches_on_ring4(self):
        import torch.multiprocessing as mp

        workdir = tempfile.mkdtemp(prefix="trim_b_usp_sdpa_")
        self.addCleanup(shutil.rmtree, workdir, ignore_errors=True)
        _write_usp_workdir(workdir)
        mp.spawn(
            _usp_worker_forced_sdpa,
            args=(WORLD_SIZE, 29886, workdir),
            nprocs=WORLD_SIZE,
            join=True,
        )
        with open(os.path.join(workdir, "results.json")) as fh:
            gathered = json.load(fh)
        for case in ("allones", "boundary", "tailonly"):
            for rank, res in enumerate(gathered):
                for j, (a, b) in enumerate(zip(res[case]["off"], res[case]["on"])):
                    tol = 5e-4 if case == "allones" else max(1e-3 * abs(a), 2e-4)
                    self.assertLessEqual(
                        abs(a - b),
                        tol,
                        msg=f"sdpa-core {case} rank{rank} step{j}: off={a} on={b}",
                    )

    def test_backbone_trim_matches_on_ring4(self):
        import torch.multiprocessing as mp

        workdir = tempfile.mkdtemp(prefix="trim_b_usp_")
        self.addCleanup(shutil.rmtree, workdir, ignore_errors=True)
        _write_usp_workdir(workdir)
        mp.spawn(
            _usp_worker,
            args=(WORLD_SIZE, 29885, workdir),
            nprocs=WORLD_SIZE,
            join=True,
        )
        with open(os.path.join(workdir, "results.json")) as fh:
            gathered = json.load(fh)
        for case in ("allones", "boundary", "tailonly"):
            for rank, res in enumerate(gathered):
                off, on = res[case]["off"], res[case]["on"]
                self.assertEqual(len(off), TTT)
                for j, (a, b) in enumerate(zip(off, on)):
                    tol = 2e-4 if case == "allones" else max(1e-3 * abs(a), 1e-4)
                    self.assertLessEqual(
                        abs(a - b),
                        tol,
                        msg=f"{case} rank{rank} step{j}: off={a} on={b}",
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
