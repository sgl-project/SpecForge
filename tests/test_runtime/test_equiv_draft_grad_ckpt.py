# coding=utf-8
"""Equivalence gate for ``--draft-gradient-checkpointing`` (TTT unroll).

Recomputing each TTT step during backward must not change what the model
learns. Naive ``torch.utils.checkpoint`` around a step is incorrect here rather
than a speed tradeoff: backward replays the forward, and a replay that appends
to a shared KV cache appends a second time, corrupting every later step. So the
bar is bitwise, not a tolerance.

The test measures the backward noise floor first (two identical unpatched runs
on the same inputs) and then requires the checkpointed-vs-unpatched gradient
difference to sit at or below that floor. It also asserts the retained graph
actually shrinks, which is the point of the flag.

Tiny synthetic draft head, random weights, no model download. GPU-only.
"""

import os
import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()

SEQ = 64
TTT = 4
BACKEND = "flex_attention"


@unittest.skipUnless(CUDA, "TTT gradient-checkpointing equivalence requires CUDA")
class TestEquivDraftGradCkpt(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True, warn_only=True)
        from specforge import AutoDraftModelConfig, AutoEagle3DraftModel
        from tests.test_runtime import _fixtures as fx

        self.fx = fx
        workdir = tempfile.mkdtemp(prefix="equiv_grad_ckpt_")
        cfg = fx.write_draft_config(os.path.join(workdir, "draft.json"))
        vocab_path = fx.write_vocab_mapping(os.path.join(workdir, "vm.pt"))

        self.draft = AutoEagle3DraftModel.from_config(
            AutoDraftModelConfig.from_file(cfg),
            attention_backend=BACKEND,
            torch_dtype=torch.bfloat16,
        ).cuda()
        self.draft.load_vocab_mapping(vocab_path)
        self.draft.freeze_embedding()
        self.draft.train()

        # One set of inputs, reused byte-for-byte by every run.
        g = torch.Generator().manual_seed(1234)
        H, V = fx.H, fx.V
        loss_mask = torch.ones(1, SEQ, dtype=torch.long)
        loss_mask[:, :8] = 0  # a masked prefix, like a real prompt
        self.inputs = dict(
            input_ids=torch.randint(0, V, (1, SEQ), generator=g).cuda(),
            attention_mask=torch.ones(1, SEQ, dtype=torch.long).cuda(),
            loss_mask=loss_mask.unsqueeze(-1).cuda(),
            hidden_states=torch.randn(1, SEQ, H * 3, generator=g)
            .to(torch.bfloat16)
            .cuda(),
            target=torch.randn(1, SEQ, V, generator=g).to(torch.bfloat16).cuda(),
        )

    def _run(self, gradient_checkpointing):
        """One fwd+bwd. Returns (losses, acceptance rates, grads, retained MiB).

        ``retained`` is what the forward *added* and held until backward, so it
        is measured as a delta against a settled allocator, not as an absolute.
        """
        from specforge import OnlineEagle3Model

        model = OnlineEagle3Model(
            draft_model=self.draft,
            length=TTT,
            attention_backend=BACKEND,
            gradient_checkpointing=gradient_checkpointing,
        ).cuda()
        model.train()
        self.draft.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated()

        plosses, acceptance_rates, *_ = model(
            **{k: v.clone() for k, v in self.inputs.items()}
        )
        retained = (torch.cuda.memory_allocated() - before) / 2**20
        loss = sum(0.8**i * plosses[i] for i in range(len(plosses)))
        loss.backward()

        grads = {
            n: p.grad.detach().float().cpu()
            for n, p in self.draft.named_parameters()
            if p.grad is not None
        }
        return (
            [x.item() for x in plosses],
            [x.item() for x in acceptance_rates],
            grads,
            retained,
        )

    @staticmethod
    def _max_grad_diff(a, b):
        """Largest absolute elementwise gradient difference, and where."""
        worst, name = 0.0, ""
        for key in a:
            d = (a[key] - b[key]).abs().max().item()
            if d > worst:
                worst, name = d, key
        return worst, name

    def test_equiv_draft_grad_ckpt(self):
        base_losses, base_acc, base_grads, base_retained = self._run(False)
        self.assertTrue(
            all(torch.isfinite(torch.tensor(base_losses))),
            f"unpatched run produced non-finite losses: {base_losses}",
        )

        # Backward noise floor: the same unpatched path, run twice.
        _, _, repeat_grads, _ = self._run(False)
        floor, floor_name = self._max_grad_diff(base_grads, repeat_grads)

        ckpt_losses, ckpt_acc, ckpt_grads, ckpt_retained = self._run(True)

        self.assertEqual(
            base_losses,
            ckpt_losses,
            "per-step TTT losses changed under gradient checkpointing",
        )
        self.assertEqual(
            base_acc,
            ckpt_acc,
            "per-step acceptance rates changed under gradient checkpointing",
        )
        self.assertEqual(
            set(base_grads),
            set(ckpt_grads),
            "different parameters received gradients under checkpointing",
        )

        diff, diff_name = self._max_grad_diff(base_grads, ckpt_grads)
        self.assertLessEqual(
            diff,
            floor,
            f"gradient diff {diff:.3e} ({diff_name}) exceeds the backward "
            f"noise floor {floor:.3e} ({floor_name}): the replay is not "
            f"reproducing the forward",
        )

        print(
            f"\n[grad-ckpt equiv] noise floor {floor:.3e} | "
            f"ckpt-vs-base grad diff {diff:.3e} | "
            f"retained {base_retained:.1f} -> {ckpt_retained:.1f} MiB"
        )

        # The savings scale with seq x draft_vocab, so a tiny fixture understates
        # them; 25% is a floor with headroom, not the expected figure.
        self.assertLess(
            ckpt_retained,
            base_retained * 0.75,
            f"checkpointing did not meaningfully reduce the retained graph "
            f"({base_retained:.1f} -> {ckpt_retained:.1f} MiB)",
        )

    def test_rejects_usp_backend(self):
        """``usp`` runs collectives inside the step, which is not replay-safe."""
        from specforge import OnlineEagle3Model

        with self.assertRaises(ValueError):
            OnlineEagle3Model(
                draft_model=self.draft,
                length=TTT,
                attention_backend="usp",
                gradient_checkpointing=True,
            )


if __name__ == "__main__":
    unittest.main()
