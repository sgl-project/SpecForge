# coding=utf-8
"""M4 gate: extraction correctness vs an independent HF reference (fp tolerance).

The SGLangAdapter's extracted aux hidden states (at the recorded layer IDs) must
equal an independent HF ``output_hidden_states=True`` forward at those layers,
and the target logits must match a direct forward — within a documented bf16
tolerance. This is the only M4 numerical gate at fp tolerance (different engine),
NOT bit-exact. Also asserts the capture assertion fires on a layer mismatch.

GPU-only. Run on the H200 box via rcli.
"""

import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()
# Documented tolerance: two bf16 forward passes of the same tiny model.
RTOL, ATOL = 2e-2, 2e-2


@unittest.skipUnless(CUDA, "extraction correctness requires CUDA")
class TestExtractionVsHFReference(unittest.TestCase):
    def test_extraction_vs_hf_reference(self):
        torch.manual_seed(0)
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29564")

        from specforge.runtime.contracts import PromptTask
        from specforge.runtime.inference.capture import CaptureConfig, verify_capture
        from specforge.runtime.inference.sglang_adapter import SGLangAdapter

        H, V, SEQ = fx.H, fx.V, 12
        workdir = tempfile.mkdtemp(prefix="extract_")
        target, target_dir, aux_ids = fx.build_hf_target(
            workdir, hidden=H, layers=8, vocab=V
        )

        adapter = SGLangAdapter(target, device="cuda")
        capture = CaptureConfig.from_strategy(
            required_features={
                "input_ids",
                "attention_mask",
                "loss_mask",
                "hidden_state",
                "target",
            },
            aux_hidden_state_layer_ids=tuple(aux_ids),
            target_repr="logits",
            target_hidden_size=H,
            target_vocab_size=V,
        )

        torch.manual_seed(7)
        ids = torch.randint(0, V, (SEQ,)).tolist()
        task = PromptTask(
            task_id="t0",
            run_id="r",
            source_id="s",
            payload={"input_ids": ids, "loss_mask": [1] * SEQ},
            max_length=SEQ,
        )
        feats = adapter.generate_features([task], capture=capture)[0]
        recorded = feats.pop("__aux_layer_ids__")
        verify_capture(feats, capture, sample_id="t0", recorded_aux_layer_ids=recorded)
        self.assertEqual(tuple(recorded), tuple(aux_ids))
        self.assertEqual(feats["hidden_state"].shape[-1], 3 * H)

        # independent HF reference forward with output_hidden_states
        input_ids = torch.tensor([ids], device="cuda")
        ref = target.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            output_hidden_states=True,
            use_cache=False,
        )
        # hidden_states[i+1] == output of decoder layer i
        extracted = feats["hidden_state"].float()
        for k, layer_id in enumerate(aux_ids):
            ref_layer = ref.hidden_states[layer_id + 1].float()  # (1, SEQ, H)
            got = extracted[..., k * H : (k + 1) * H]
            torch.testing.assert_close(
                got,
                ref_layer,
                rtol=RTOL,
                atol=ATOL,
                msg=f"aux layer {layer_id} extraction drift",
            )

        # target logits match a direct forward (adapter pads next-token; compare
        # the overlapping positions [0..SEQ-2] against ref logits [1..SEQ-1])
        adapter_target = feats["target"].float()
        ref_logits = ref.logits.float()
        torch.testing.assert_close(
            adapter_target[:, : SEQ - 1, :],
            ref_logits[:, 1:SEQ, :],
            rtol=RTOL,
            atol=ATOL,
            msg="target logits drift vs direct lm_head",
        )

    def test_capture_layer_mismatch_fails(self):
        """A recorded aux-layer set != requested fails loudly at the boundary."""
        from specforge.runtime.inference.capture import (
            CaptureConfig,
            CaptureMismatchError,
            verify_capture,
        )

        cap = CaptureConfig.from_strategy(
            required_features={"hidden_state", "target", "input_ids", "loss_mask"},
            aux_hidden_state_layer_ids=(1, 3, 4),
            target_repr="hidden_state",
            target_hidden_size=8,
        )
        tensors = {
            "input_ids": torch.zeros(1, 4, dtype=torch.long),
            "loss_mask": torch.ones(1, 4, dtype=torch.long),
            "hidden_state": torch.randn(1, 4, 24),
            "target": torch.randn(1, 4, 8),
        }
        with self.assertRaises(CaptureMismatchError):
            verify_capture(
                tensors, cap, sample_id="x", recorded_aux_layer_ids=(1, 3, 5)
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
