# coding=utf-8
"""Equivalence: trim_loss_positions must not change the training loss.

A-level position trimming computes the teacher target_p, the draft logits and the
loss only at supervised (loss-masked) positions instead of over the full sequence.
It is mathematically equivalent to the full-length path (the mean denominator is
rescaled from n_sup back to the full length). This test runs the identical forward
with trimming off and on and asserts the per-step losses match within bf16
tolerance.

GPU-only, matching the other EAGLE3 equivalence tests in this directory.
"""

import os
import shutil
import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "trim_loss_positions equivalence requires CUDA")
class TestEquivTrimLossPositions(unittest.TestCase):
    def test_trim_loss_positions_matches_full(self):
        torch.manual_seed(0)
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29567")

        workdir = tempfile.mkdtemp(prefix="equiv_trim_")
        self.addCleanup(shutil.rmtree, workdir, ignore_errors=True)

        model, target_head = fx.build_eagle3(workdir, ttt=3)
        model.eval()

        # One offline sample gives us (input_ids, target, loss_mask, hidden_state).
        feature_dir = os.path.join(workdir, "features")
        os.makedirs(feature_dir, exist_ok=True)
        fx.write_offline_files(feature_dir, n=1, seq=16)
        batch = torch.load(
            os.path.join(feature_dir, sorted(os.listdir(feature_dir))[0]),
            map_location="cpu",
        )

        # `hidden_state` is the target-side capture that the target head turns into
        # the teacher distribution; `aux_hidden_state` is the draft backbone input.
        input_ids, target, loss_mask = target_head.preprocess(
            batch["input_ids"].unsqueeze(0),
            batch["hidden_state"],
            batch["loss_mask"].unsqueeze(0),
        )
        target = target_head(target.cuda())
        hidden_states = batch["aux_hidden_state"].cuda()
        input_ids = input_ids.cuda()

        # Prompt-heavy mask so trimming is non-trivial: first half unsupervised.
        loss_mask = loss_mask.cuda().clone()
        loss_mask[:, : loss_mask.shape[1] // 2] = 0
        attention_mask = torch.ones_like(input_ids)

        @torch.no_grad()
        def step_losses(trim: bool):
            model.trim_loss_positions = trim
            plosses, *_ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                target=target,
                hidden_states=hidden_states,
            )
            return [float(p.item()) for p in plosses]

        full = step_losses(False)
        trimmed = step_losses(True)

        self.assertEqual(len(full), len(trimmed))
        for i, (a, b) in enumerate(zip(full, trimmed)):
            tol = 5e-3 * max(abs(a), abs(b)) + 1e-4
            self.assertLessEqual(
                abs(a - b),
                tol,
                msg=f"step {i}: full={a} trimmed={b} (tol={tol})",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
