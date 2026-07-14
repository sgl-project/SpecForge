# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Target-capture stability gates for the canonical runtime path.

The runtime has one target-engine API, ``capture()``. These GPU tests pin two
properties directly on that path:

1. A TrainBatch-style digest over captured tensors is stable run-to-run with a
   fixed seed.
2. Per-step loss through online capture→train is reproducible at ``atol=1e-4``.

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
class TargetCaptureStabilityTest(unittest.TestCase):
    """The canonical capture path emits deterministic training features."""

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
        from specforge.core.eagle3 import OnlineEagle3Model
        from specforge.launch import build_online_runtime
        from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
        from specforge.optimizer import BF16Optimizer
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port=port)
        H, V, SEQ, TTT, ACC, MAX_OPT_STEPS, N = fx.H, fx.V, 12, 3, 2, 2, 8
        workdir = tempfile.mkdtemp(prefix="pb_gate_loss_")

        target, _dir, aux_ids = fx.build_hf_target(workdir, hidden=H, layers=8, vocab=V)
        cfg = fx.write_draft_config(os.path.join(workdir, "draft.json"))
        vocab_path = fx.write_vocab_mapping(os.path.join(workdir, "vm.pt"))
        torch.manual_seed(0)
        draft = AutoDraftModel.from_config(
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

        trainer, loader, workers, controller, run_interleaved = build_online_runtime(
            strategy="eagle3",
            target_model=target,
            prompts=prompts,
            draft_model=eagle3_model,
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
            max_steps=MAX_OPT_STEPS,
        )
        # capture per-optimizer-step loss via the controller's logger hook
        trainer.logger = lambda metrics, step: losses.append(metrics["loss"])
        trainer.log_interval = 1
        run_interleaved()
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
