# coding=utf-8
"""Phase C gate: the control-plane mode must not change the training.

Trains the same eagle3 draft over the same offline features twice, varying only
the DeploymentMode: once through the lightweight ``local_colocated`` plane (no-op
metadata store, no durable ack) and once through the full ``disaggregated`` plane
(durable store + optimizer-boundary ack). The per-step loss curves must be
bit-identical — colocated pays nothing for the control plane it does not use, and
gets exactly the same result.

Only the control plane differs; the data plane (LocalFeatureStore over the same
file:// features) and the model weights are held constant, isolating the axis
Phase C changes. GPU-only. Run on the H200 box via rcli.
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

        from specforge.optimizer import BF16Optimizer
        from specforge.runtime.control_plane import resolve_control_plane
        from specforge.runtime.data_plane import LocalFeatureStore
        from specforge.launch import _offline_io
        from specforge.training.strategies.registry import resolve_strategy
        from specforge.training import Trainer

        TTT, N, MAX_LEN = 3, 8, 512
        workdir = tempfile.mkdtemp(prefix="coloc_disagg_")
        feat_dir = fx.write_offline_files(os.path.join(workdir, "features"), n=N)

        def run(mode: str):
            # identical model weights + identical features on every run
            torch.manual_seed(0)
            torch.use_deterministic_algorithms(True, warn_only=True)
            eagle3_model, target_head = fx.build_eagle3(workdir, ttt=TTT)

            spec = resolve_strategy("eagle3")
            refs = spec.make_offline_reader(
                feat_dir, run_id=mode, ttt_length=TTT, max_len=MAX_LEN
            ).read()
            store = LocalFeatureStore(mode)
            controller, durable_ack = resolve_control_plane(mode, mode)
            collate_fn, per_sample_transform = _offline_io(spec, MAX_LEN)

            losses = []
            trainer = Trainer(
                spec=spec,
                controller=controller,
                store=store,
                ref_source={"refs": refs},
                model=eagle3_model,
                target_head=target_head,
                optimizer_factory=lambda m: BF16Optimizer(
                    m, lr=1e-3, max_grad_norm=0.5, warmup_ratio=0.0, total_steps=10
                ),
                run_id=mode,
                output_dir=os.path.join(workdir, mode),
                batch_size=1,
                accumulation_steps=1,
                num_epochs=1,
                max_steps=None,
                save_interval=0,
                eval_interval=0,
                tp_size=1,
                sp_ulysses_size=1,
                sp_ring_size=1,
                logger=lambda metrics, step: losses.append(metrics["loss"]),
                log_interval=1,
                collate_fn=collate_fn,
                per_sample_transform=per_sample_transform,
                durable_ack=durable_ack,
            )
            torch.manual_seed(0)
            trainer.fit()
            return losses

        colocated = run("local_colocated")
        disagg = run("disaggregated")

        self.assertTrue(colocated, "no steps ran")
        self.assertEqual(len(colocated), len(disagg))
        for i, (a, b) in enumerate(zip(colocated, disagg)):
            self.assertEqual(a, b, msg=f"loss diverged at step {i}: {a} vs {b}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
