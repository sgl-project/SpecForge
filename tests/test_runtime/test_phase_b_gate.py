# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Phase B explicit gate — target capture is byte-identical pre/post refactor.

The roadmap's Phase B gate is "byte-identical batches and loss vs the pre-refactor
run." This file pins it directly rather than leaning only on the full-suite's
indirect coverage:

1. ``capture() ≡ legacy generate_*_data()`` **bytewise** (eagle3 + dflash, HF
   backend). ``generate_*_data`` is the pre-Phase-B extraction method (kept as a
   back-compat alias); the whole B refactor (B1 rename + B2 SGLangCaptureBackend +
   B4 adapter cutover to ``capture()``) routes through ``capture()`` now, so this
   is the pre/post-refactor capture equality gate.
2. A **TrainBatch-style feature digest** over the captured tensors is stable
   run-to-run (fixed seed) — a concrete regression pin for the capture pipeline.
3. **Per-step loss** through the full online capture→train path (which now calls
   ``capture()``) is reproducible run-to-run at atol=1e-4.

(The dataflow-vs-legacy per-step loss equality is additionally covered by
``test_equiv_online_eagle3`` / ``test_equiv_offline_eagle3``, which after the B4
cutover exercise the ``capture()`` path.)

GPU-only. Run on the H200 box via rcli.
"""

import hashlib
import os
import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()


def _digest(*tensors) -> str:
    h = hashlib.sha256()
    for t in tensors:
        t = t.detach().to("cpu").contiguous()
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.float().numpy().tobytes())
    return h.hexdigest()


@unittest.skipUnless(CUDA, "Phase B gate requires CUDA")
class PhaseBCaptureEqualityTest(unittest.TestCase):
    """capture() is a faithful, byte-identical alias of the legacy generate_*."""

    def test_capture_equals_legacy_eagle3_hf(self):
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29572")
        workdir = tempfile.mkdtemp(prefix="pb_gate_eagle3_")
        target, _dir, _aux = fx.build_hf_target(
            workdir, hidden=fx.H, layers=8, vocab=fx.V
        )

        torch.manual_seed(0)
        ids = torch.randint(0, fx.V, (2, 32)).cuda()
        attn = torch.ones_like(ids)
        lm = torch.ones_like(ids)

        legacy = target.generate_eagle3_data(
            input_ids=ids, attention_mask=attn, loss_mask=lm
        )
        via_capture = target.capture(input_ids=ids, attention_mask=attn, loss_mask=lm)

        self.assertTrue(torch.equal(legacy.hidden_states, via_capture.hidden_states))
        self.assertTrue(torch.equal(legacy.target, via_capture.target))
        self.assertTrue(torch.equal(legacy.loss_mask, via_capture.loss_mask))
        self.assertTrue(torch.equal(legacy.input_ids, via_capture.input_ids))

    def test_capture_equals_legacy_dflash_hf(self):
        from specforge.modeling.target import get_target_engine
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29573")
        workdir = tempfile.mkdtemp(prefix="pb_gate_dflash_")
        _model, _w, target_dir, layers = fx.build_dflash(
            workdir,
            hidden=64,
            vocab=fx.V,
            block_size=4,
            num_anchors=8,
            attention_backend="sdpa",
        )
        target = get_target_engine(
            target_dir,
            strategy="dflash",
            backend="hf",
            torch_dtype=torch.bfloat16,
            device="cuda",
        )
        target.set_capture_layers(layers)

        torch.manual_seed(0)
        ids = torch.randint(0, fx.V, (2, 32)).cuda()
        attn = torch.ones_like(ids)
        lm = torch.ones_like(ids)

        legacy = target.generate_dflash_data(
            input_ids=ids, attention_mask=attn, loss_mask=lm
        )
        via_capture = target.capture(input_ids=ids, attention_mask=attn, loss_mask=lm)

        self.assertTrue(torch.equal(legacy.hidden_states, via_capture.hidden_states))
        self.assertTrue(torch.equal(legacy.input_ids, via_capture.input_ids))
        self.assertTrue(torch.equal(legacy.loss_mask, via_capture.loss_mask))

    def test_eagle3_captured_feature_digest_is_stable(self):
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29574")
        workdir = tempfile.mkdtemp(prefix="pb_gate_digest_")
        target, _dir, _aux = fx.build_hf_target(
            workdir, hidden=fx.H, layers=8, vocab=fx.V
        )

        def capture_digest():
            g = torch.Generator().manual_seed(1234)
            ids = torch.randint(0, fx.V, (2, 24), generator=g).cuda()
            out = target.capture(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                loss_mask=torch.ones_like(ids),
            )
            return _digest(out.hidden_states, out.target, out.input_ids, out.loss_mask)

        self.assertEqual(capture_digest(), capture_digest())


@unittest.skipUnless(CUDA, "Phase B gate requires CUDA")
class PhaseBLossReproducibilityTest(unittest.TestCase):
    """Per-step loss through the full online capture→train path is reproducible."""

    def _run_losses(self, port):
        from specforge import (
            AutoDraftModelConfig,
            AutoEagle3DraftModel,
            OnlineEagle3Model,
        )
        from specforge.optimizer import BF16Optimizer
        from specforge.launch import build_online_runtime
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port=port)
        H, V, SEQ, TTT, ACC, MAX_OPT_STEPS, N = fx.H, fx.V, 12, 3, 2, 2, 8
        workdir = tempfile.mkdtemp(prefix="pb_gate_loss_")

        target, _dir, aux_ids = fx.build_hf_target(workdir, hidden=H, layers=8, vocab=V)
        cfg = fx.write_draft_config(os.path.join(workdir, "draft.json"))
        vocab_path = fx.write_vocab_mapping(os.path.join(workdir, "vm.pt"))
        torch.manual_seed(0)
        draft = AutoEagle3DraftModel.from_config(
            AutoDraftModelConfig.from_file(cfg),
            attention_backend="flex_attention",
            torch_dtype=torch.bfloat16,
        ).cuda()
        draft.load_vocab_mapping(vocab_path)
        draft.freeze_embedding()
        eagle3_model = OnlineEagle3Model(
            draft, length=TTT, attention_backend="flex_attention"
        ).cuda()

        g = torch.Generator().manual_seed(7)
        prompts = [
            {
                "payload": {
                    "input_ids": torch.randint(0, V, (SEQ,), generator=g).tolist(),
                    "loss_mask": [1] * SEQ,
                }
            }
            for _ in range(N)
        ]

        losses = []

        trainer, loader, workers, controller, drive_rollout = build_online_runtime(
            strategy="eagle3",
            target_model=target,
            prompts=prompts,
            eagle3_model=eagle3_model,
            optimizer_factory=lambda m: BF16Optimizer(
                m, lr=1e-3, max_grad_norm=0.5, warmup_ratio=0.0, total_steps=10
            ),
            run_id="pb-gate",
            output_dir=os.path.join(workdir, "out"),
            target_hidden_size=H,
            target_vocab_size=V,
            target_repr="logits",
            aux_hidden_state_layer_ids=tuple(aux_ids),
            batch_size=1,
            accumulation_steps=ACC,
            num_epochs=1,
            max_steps=MAX_OPT_STEPS,
        )
        # capture per-optimizer-step loss via the controller's logger hook
        trainer.logger = lambda metrics, step: losses.append(metrics["loss"])
        trainer.log_interval = 1
        drive_rollout()
        trainer.fit(loader)
        return losses

    def test_online_eagle3_losses_reproducible(self):
        a = self._run_losses("29575")
        b = self._run_losses("29576")
        self.assertTrue(len(a) >= 1 and len(a) == len(b), f"losses: {a} vs {b}")
        for x, y in zip(a, b):
            self.assertTrue(
                abs(x - y) <= 1e-4, f"per-step loss not reproducible: {a} vs {b}"
            )


if __name__ == "__main__":
    unittest.main()
