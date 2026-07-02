# coding=utf-8
"""Phase C gate: the control-plane mode must not change the training.

Trains the same eagle3 draft over the same offline features twice **through the
same builder** (``build_offline_runtime``), varying only ``deployment_mode``:
once through the lightweight ``local_colocated`` plane (no-op metadata store, no
durable ack) and once through the full ``disaggregated`` plane (SQLite durable
store + optimizer-boundary ack). The per-step loss curves must be bit-identical
— colocated pays nothing for the control plane it does not use, and gets exactly
the same result.

Only the control plane differs; the data plane (LocalFeatureStore over the same
file:// features) and the model weights are held constant, isolating the axis
Phase C changes. The disagg leg additionally asserts the durable SQLite marker
was really written (the heavy plane was exercised, not silently bypassed).
GPU-only. Run on the H200 box via rcli.
"""

import os
import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "colocated/disagg equivalence requires CUDA")
class TestColocatedVsDisaggEquiv(unittest.TestCase):
    def test_loss_curve_independent_of_control_plane(self):
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29571")

        from specforge.launch import build_offline_runtime
        from specforge.optimizer import BF16Optimizer
        from specforge.runtime.control_plane.metadata_store import SQLiteMetadataStore

        TTT, N, MAX_LEN = 3, 8, 512
        workdir = tempfile.mkdtemp(prefix="coloc_disagg_")
        feat_dir = fx.write_offline_files(os.path.join(workdir, "features"), n=N)

        def run(mode: str, metadata_db_path=None):
            # identical model weights + identical features on every run
            torch.manual_seed(0)
            eagle3_model, target_head = fx.build_eagle3(workdir, ttt=TTT)

            losses = []
            trainer, loader = build_offline_runtime(
                strategy="eagle3",
                hidden_states_path=feat_dir,
                eagle3_model=eagle3_model,
                target_head=target_head,
                optimizer_factory=lambda m: BF16Optimizer(
                    m, lr=1e-3, max_grad_norm=0.5, warmup_ratio=0.0, total_steps=10
                ),
                run_id=mode,
                output_dir=os.path.join(workdir, mode),
                ttt_length=TTT,
                max_len=MAX_LEN,
                batch_size=1,
                logger=lambda metrics, step: losses.append(metrics["loss"]),
                log_interval=1,
                deployment_mode=mode,
                metadata_db_path=metadata_db_path,
            )
            torch.manual_seed(0)
            trainer.fit(loader)
            return losses

        # deterministic kernels for the bit-identity claim; restore afterwards so
        # the flag does not leak into other tests in the same process.
        prev_det = torch.are_deterministic_algorithms_enabled()
        prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            colocated = run("local_colocated")
            db_path = os.path.join(workdir, "disagg.sqlite")
            disagg = run("disaggregated", metadata_db_path=db_path)
        finally:
            torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)

        self.assertTrue(colocated, "no steps ran")
        self.assertEqual(len(colocated), len(disagg))
        for i, (a, b) in enumerate(zip(colocated, disagg)):
            self.assertEqual(a, b, msg=f"loss diverged at step {i}: {a} vs {b}")

        # The disagg leg must have gone through the durable plane: the SQLite
        # marker records the optimizer-boundary ack transaction.
        store = SQLiteMetadataStore(db_path)
        try:
            marker = store.durable_marker()
        finally:
            store.close()
        self.assertTrue(marker["acked"], "disagg leg recorded no durable acks")
        self.assertTrue(marker["optimizer_durable"])
        self.assertEqual(marker["global_step"], len(disagg))


if __name__ == "__main__":
    unittest.main(verbosity=2)
