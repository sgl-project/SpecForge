# coding=utf-8
"""Canonical trim/full loss parity on one GPU and four-rank USP.

The single-GPU gate checks the closed-form zero-logit loss. The distributed
gate compares supervised, boundary and overlap-tail masks across four ranks.
"""

import json
import math
import os
import shutil
import tempfile
import unittest

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


@unittest.skipUnless(CUDA, "loss kernel is a Triton kernel")
class TestTrimLossAnalytic(unittest.TestCase):
    """Zero logits + normalized teachers => masked row loss == ln(D) exactly."""

    def test_trim_and_full_hit_closed_form(self):
        from specforge.losses.soft_cross_entropy import LogSoftmaxLoss

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
        from dataclasses import replace

        import torch.distributed as dist

        from specforge.algorithms.objectives import PredictionBatch
        from specforge.algorithms.objectives.execution import execute_batch
        from specforge.runtime.data_plane import FeatureDataLoader, LocalFeatureStore
        from tests.test_algorithms.fixtures import builtin_algorithm_registry

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.use_deterministic_algorithms(True, warn_only=True)
        model = fx.load_eagle3(
            os.path.join(workdir, "draft.json"),
            os.path.join(workdir, "target"),
            os.path.join(workdir, "vocab_mapping.pt"),
            backend="usp",
            ttt=TTT,
        )
        model.train()
        algorithm = builtin_algorithm_registry(ttt_length=TTT).resolve("eagle3")
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
                tensors = {key: value.cuda() for key, value in batch.tensors.items()}
                execution = replace(model.execution, trim=trim)
                with torch.no_grad():
                    predictions = model.objective.rollout(model, tensors)
                    return [
                        execute_batch(
                            PredictionBatch([prediction], reduction="sum_means"),
                            None,
                            model.weighting,
                            training=True,
                            execution=execution,
                            selection=model.loss_implementations,
                        ).loss.item()
                        for prediction in predictions
                    ]

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
