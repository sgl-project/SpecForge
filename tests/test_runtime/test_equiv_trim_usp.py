# coding=utf-8
"""trim_loss_positions under USP sequence parallelism.

Three layers, so CI keeps guarding the USP row-selection math even on hosts
without four GPUs:

1. ``TestTrimPackGolden`` (CPU) -- hand-derived literal tables for the per-step
   row sets: the overlap-tail bound (``s - j < chunk_len``), dead-step padding,
   the unreachable-supervision fallback, and non-USP back-compat.
2. ``TestTrimLossAnalytic`` (one GPU) -- with zero logits and normalized
   teachers every masked row's loss is exactly ``ln(draft_vocab)``, so the
   trim-scaled and full-shaped losses must both hit a closed-form constant.
3. ``TestEquivTrimUspFourRank`` (four GPUs + flash-attn, like
   ``test_equiv_4rank``) -- per-step loss parity, trim ON vs OFF, on a real
   ring-4 offline pipeline across three adversarial masks: all-supervised
   (trivial-equality boundary: any row/denominator error is exposed without
   tolerance cover), supervision straddling every rank boundary
   (overlap-tail-as-teacher), and supervision only inside one rank's overlap
   tail (dead steps plus ranks with no supervision at all, which fall back to
   the full path -- the mixed-path collective-alignment hazard).
"""

import json
import math
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


class TestTrimPackGolden(unittest.TestCase):
    """Hand-derived expected outputs for _build_trim_pack (CPU only)."""

    def _mk(self, mask_list, seed=1, vocab=32, draft=8):
        g = torch.Generator().manual_seed(seed)
        ids = torch.randperm(vocab, generator=g)[:draft].sort().values
        t2d = torch.zeros(vocab, dtype=torch.bool)
        t2d[ids] = True
        L = len(mask_list)
        lm = torch.tensor(mask_list, dtype=torch.long).view(1, L, 1)
        tgt = torch.randn(1, L, vocab, generator=g)
        return tgt, t2d, lm

    def test_usp_tail_bound(self):
        # C=6, ttt=2, local_len=8, supervised {4,5,6}; 6 is an overlap-tail
        # position: a legal teacher, never a loss row.
        #   step0: {s : s-0 < 6} = {4,5}   -> rows [4,5],   n=2
        #   step1: all of {4,5,6}          -> rows [3,4,5], n=3
        from specforge.algorithms.eagle3.model import _build_trim_pack

        tgt, t2d, lm = self._mk([0, 0, 0, 0, 1, 1, 1, 0])
        p = _build_trim_pack(tgt, t2d, lm, length=2, chunk_len=6)
        self.assertEqual(p["full_len"], 6)
        self.assertEqual(p["sup"].tolist(), [4, 5, 6])
        self.assertEqual(p["rows_steps"][0].tolist(), [4, 5])
        self.assertEqual(p["keep_steps"][0].tolist(), [0, 1])
        self.assertEqual(p["nrows_steps"][0], 2)
        self.assertEqual(p["rows_steps"][1].tolist(), [3, 4, 5])
        self.assertEqual(p["nrows_steps"][1], 3)

    def test_usp_dead_step(self):
        # Supervision only at {6,7} (pure tail). Position 7 can never reach a
        # row (needs j >= 2) and is filtered; 6 is unreachable at step 0.
        from specforge.algorithms.eagle3.model import _build_trim_pack

        tgt, t2d, lm = self._mk([0, 0, 0, 0, 0, 0, 1, 1])
        p = _build_trim_pack(tgt, t2d, lm, length=2, chunk_len=6)
        self.assertEqual(p["sup"].tolist(), [6])
        self.assertEqual(p["nrows_steps"][0], 0)  # dead step
        self.assertEqual(p["rows_steps"][0].tolist(), [0])  # padded dummy
        self.assertEqual(p["rows_steps"][1].tolist(), [5])
        self.assertEqual(p["nrows_steps"][1], 1)

    def test_unreachable_supervision_falls_back(self):
        from specforge.algorithms.eagle3.model import _build_trim_pack

        tgt, t2d, lm = self._mk([0, 0, 0, 0, 0, 0, 0, 1])
        self.assertIsNone(_build_trim_pack(tgt, t2d, lm, length=2, chunk_len=6))

    def test_empty_supervision_falls_back(self):
        from specforge.algorithms.eagle3.model import _build_trim_pack

        tgt, t2d, lm = self._mk([0] * 8)
        self.assertIsNone(_build_trim_pack(tgt, t2d, lm, length=2, chunk_len=6))

    def test_reuses_position_mask_from_teacher_computation(self):
        from specforge.algorithms.eagle3 import model as eagle_model

        tgt, t2d, lm = self._mk([0, 1, 0, 1])
        nrows = 2
        draft_vocab = int(t2d.sum())
        teacher = (
            torch.zeros(1, nrows, draft_vocab),
            torch.zeros(1, nrows, draft_vocab),
            torch.zeros(1, nrows, dtype=torch.long),
            torch.full((1, nrows, 1), 7),
        )
        with mock.patch.object(
            eagle_model, "_compute_target_p_eager", return_value=teacher
        ):
            pack = eagle_model._build_trim_pack(tgt, t2d, lm, length=2)

        self.assertIs(pack["position_mask_sup"], teacher[3])

    def test_non_usp_backcompat(self):
        # chunk_len=None -> C=L: the pre-USP semantics, plus full_len now comes
        # from the row count rather than the (possibly padded) mask length.
        from specforge.algorithms.eagle3.model import _build_trim_pack

        tgt, t2d, lm = self._mk([1, 0, 0, 0, 1, 0, 0, 1])
        p = _build_trim_pack(tgt, t2d, lm, length=2)
        self.assertEqual(p["full_len"], 8)
        self.assertEqual(p["rows_steps"][0].tolist(), [0, 4, 7])
        self.assertEqual(p["rows_steps"][1].tolist(), [3, 6])
        self.assertEqual(p["keep_steps"][1].tolist(), [1, 2])
        self.assertEqual(p["nrows_steps"][1], 2)


class TestTrimAdapterViews(unittest.TestCase):
    def _inputs(self):
        return {
            "global_input_ids": torch.arange(8).view(1, 8),
            "hidden_states": torch.arange(24).view(1, 8, 3),
            "attention_mask": torch.ones(1, 8),
            "position_ids": torch.arange(16).view(1, 16),
        }

    def test_default_adapter_keeps_full_backbone_view(self):
        from specforge.core.eagle3_adapters import BackendAdapter

        adapter = BackendAdapter(model=None)
        inputs = self._inputs()
        self.assertEqual(adapter.backbone_row_count(seq_length=8, ttt_length=2), 8)

        view = adapter.backbone_view(row_count=8, **inputs)

        self.assertIs(view.input_ids, inputs["global_input_ids"])
        self.assertIs(view.hidden_states, inputs["hidden_states"])
        self.assertIs(view.attention_mask, inputs["attention_mask"])
        self.assertIs(view.position_ids, inputs["position_ids"])

    def test_usp_adapter_owns_chunk_and_position_slicing(self):
        from specforge.core import eagle3_adapters

        world_sizes = {"sp": 4, "ulysses": 2}
        with (
            mock.patch.object(eagle3_adapters, "get_draft_sp_group", return_value="sp"),
            mock.patch.object(
                eagle3_adapters, "get_sp_ulysses_group", return_value="ulysses"
            ),
            mock.patch.object(
                eagle3_adapters.dist,
                "get_world_size",
                side_effect=lambda group: world_sizes[group],
            ),
        ):
            adapter = eagle3_adapters.UspAdapter(model=None)

        inputs = self._inputs()
        row_count = adapter.backbone_row_count(seq_length=8, ttt_length=2)
        view = adapter.backbone_view(row_count=row_count, **inputs)

        self.assertEqual(row_count, 6)
        self.assertEqual(view.input_ids.shape, (1, 6))
        self.assertEqual(view.hidden_states.shape, (1, 6, 3))
        self.assertEqual(view.attention_mask.shape, (1, 6))
        self.assertEqual(view.position_ids.shape, (1, 12))


@unittest.skipUnless(CUDA, "loss kernel is a Triton kernel")
class TestTrimLossAnalytic(unittest.TestCase):
    """Zero logits + normalized teachers => masked row loss == ln(D) exactly."""

    def test_trim_and_full_hit_closed_form(self):
        from specforge.core.loss import LogSoftmaxLoss

        D = 64
        g = torch.Generator().manual_seed(2)

        def one_hot_rows(n):
            t = torch.zeros(1, n, D, device="cuda")
            t[0, torch.arange(n), torch.randint(0, D, (n,), generator=g)] = 1.0
            return t

        # Trim-shaped: 3 selected rows, all masked-in; kernel mean == ln(D);
        # rescaled by nrows/C = 3/6 -> 3*ln(64)/6.
        kernel = LogSoftmaxLoss.apply(
            torch.zeros(1, 3, D, device="cuda"),
            one_hot_rows(3),
            torch.ones(1, 3, 1, device="cuda"),
        )
        self.assertAlmostEqual(kernel.item(), math.log(D), places=5)
        self.assertAlmostEqual((kernel * (3 / 6)).item(), 2.0794415417, places=5)

        # Full-shaped: 6 rows, 3 masked-in -> same constant with no rescale.
        pm = torch.tensor([0, 1, 0, 1, 1, 0], device="cuda").view(1, 6, 1)
        full = LogSoftmaxLoss.apply(
            torch.zeros(1, 6, D, device="cuda"), one_hot_rows(6), pm
        )
        self.assertAlmostEqual(full.item(), 2.0794415417, places=5)


def _write_workdir(workdir):
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
    return list(masks)


def _worker(rank, world_size, port, workdir):
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
                run_id=f"trimusp-{case}",
                ttt_length=TTT,
                max_len=SEQ,
            ).read()
            loader = FeatureDataLoader(
                LocalFeatureStore(f"trimusp-{case}-{rank}"),
                refs=refs,
                batch_size=1,
                collate_fn=provider.build_collator(),
                per_sample_transform=provider.build_normalizer(
                    SEQ, ttt_length=TTT, use_usp_preprocess=True
                ),
                strategy=algorithm.name,
            )
            batch = next(iter(loader))

            def step_losses(trim):
                strat = algorithm.providers.step.build(
                    model,
                    target_head=target_head,
                    trim_loss_positions=trim,
                )
                with torch.no_grad():
                    out = strat.forward_loss(batch)
                return [float(p.item()) for p in out.metrics["plosses"]]

            results[case] = {"full": step_losses(False), "trim": step_losses(True)}

        gathered = [None] * world_size
        dist.all_gather_object(gathered, results)
        if rank == 0:
            with open(os.path.join(workdir, "results.json"), "w") as fh:
                json.dump(gathered, fh)
        dist.barrier()
    finally:
        from specforge.distributed import destroy_distributed

        destroy_distributed()


@unittest.skipUnless(
    CUDA and NGPU >= WORLD_SIZE and _has_standard_flash_attention(),
    "requires four CUDA devices and the standard flash-attn USP interfaces",
)
class TestEquivTrimUspFourRank(unittest.TestCase):
    def test_trim_matches_full_per_step_on_ring4(self):
        import torch.multiprocessing as mp

        workdir = tempfile.mkdtemp(prefix="trim_usp_")
        self.addCleanup(shutil.rmtree, workdir, ignore_errors=True)
        _write_workdir(workdir)
        mp.spawn(
            _worker,
            args=(WORLD_SIZE, 29871, workdir),
            nprocs=WORLD_SIZE,
            join=True,
        )
        with open(os.path.join(workdir, "results.json")) as fh:
            gathered = json.load(fh)
        for case in ("allones", "boundary", "tailonly"):
            for rank, res in enumerate(gathered):
                full, trim = res[case]["full"], res[case]["trim"]
                self.assertEqual(len(full), TTT)
                for j, (a, b) in enumerate(zip(full, trim)):
                    if case == "allones":
                        # expected near-bit-equal; 1e-6 is ~4 orders below the
                        # smallest possible discrete error (one row's worth,
                        # ~loss/C) while allowing 1-2 ulp of fp32 noise
                        tol = 1e-6
                    else:
                        tol = max(1e-3 * abs(a), 1e-4)
                    self.assertLessEqual(
                        abs(a - b),
                        tol,
                        msg=f"{case} rank{rank} step{j}: full={a} trim={b}",
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
