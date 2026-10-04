"""Packed EAGLE3 must retain padded-batch supervision, gradients, and isolation."""

import copy
import tempfile
import unittest

import torch

from specforge.modeling.packed_sequence import (
    PackedSequenceLayout,
    generate_packed_eagle3_mask,
)


class PackedSequenceLayoutTest(unittest.TestCase):
    def test_repeated_shifts_never_import_the_next_document(self):
        layout = PackedSequenceLayout.from_lengths(torch.tensor([3, 2]), 5, "cpu")
        values = torch.tensor([[1, 2, 3, 4, 5]])
        for expected in ([2, 3, 0, 5, 0], [3, 0, 0, 0, 0], [0, 0, 0, 0, 0]):
            values = layout.shift_left(values)
            self.assertEqual(values.tolist(), [expected])

    def test_mask_matches_independent_documents_at_every_depth(self):
        lengths = [5, 2, 4]
        layout = PackedSequenceLayout.from_lengths(
            torch.tensor(lengths), sum(lengths), "cpu"
        )
        size = sum(lengths)
        for depth in range(4):
            mask = generate_packed_eagle3_mask(layout, size, depth)
            for q in range(size + 2):
                for k in range(size * (depth + 1)):
                    row = k % size
                    valid = q < size
                    if valid:
                        valid = (
                            layout.document_ids[q] == layout.document_ids[row]
                            and layout.positions[q] < layout.document_lengths[q] - depth
                            and layout.positions[row]
                            < layout.document_lengths[row] - depth
                        )
                    expected = bool(
                        valid and ((k < size and q >= k) or (k >= size and row == q))
                    )
                    actual = bool(mask(0, 0, torch.tensor(q), torch.tensor(k)))
                    self.assertEqual(actual, expected, (depth, q, k))


@unittest.skipUnless(
    torch.cuda.is_available(), "production Flex Attention and loss require CUDA"
)
class Eagle3PackedProductionTest(unittest.TestCase):
    def test_cpu_layout_can_shift_device_resident_hidden_features(self):
        layout = PackedSequenceLayout.from_lengths(torch.tensor([3, 2]), 5, "cpu")
        hidden = torch.arange(10, device="cuda").view(1, 5, 2)
        torch.testing.assert_close(
            layout.shift_left(hidden),
            torch.tensor([[[2, 3], [4, 5], [0, 0], [8, 9], [0, 0]]], device="cuda"),
        )

    def _fixtures(self, dtype=torch.float32, rope_scaling=None):
        from transformers import LlamaConfig

        from specforge.algorithms.eagle3.model import OnlineEagle3Model
        from specforge.modeling.draft.llama3_eagle import LlamaForCausalLMEagle3
        from specforge.modeling.target.target_head import TargetHead

        torch.manual_seed(123)
        config = LlamaConfig(
            vocab_size=64,
            draft_vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=160,
            rope_scaling=rope_scaling,
            pad_token_id=0,
        )
        draft = (
            LlamaForCausalLMEagle3(config, attention_backend="flex_attention")
            .cuda()
            .to(dtype)
        )
        model = OnlineEagle3Model(draft, length=4, attention_backend="flex_attention")

        # Use the real frozen head methods, without downloading any checkpoint.
        head = TargetHead.__new__(TargetHead)
        torch.nn.Module.__init__(head)
        head.fc = torch.nn.Linear(64, 64, bias=False).cuda().to(dtype)
        head.freeze_weights()
        features = []
        for length in (131, 67, 3):
            mask = torch.ones(1, length, dtype=torch.long)
            mask[:, -1] = 0
            # A prompt mask and an internal supervision gap exercise mask shifts.
            if length > 10:
                mask[:, :4] = 0
                mask[:, 9:12] = 0
            features.append(
                {
                    "input_ids": torch.randint(1, 64, (1, length)),
                    "attention_mask": torch.ones(1, length, dtype=torch.long),
                    "loss_mask": mask,
                    "hidden_state": torch.randn(1, length, 192).to(dtype),
                    "target": torch.randn(1, length, 64).to(dtype),
                }
            )
        return model, head, features

    def _run(self, model, head, features, packed):
        from specforge.algorithms.eagle3.data import DataCollatorWithPacking
        from specforge.data.utils import DataCollatorWithPadding
        from specforge.runtime.contracts import TrainBatch
        from specforge.training.strategies.base import Eagle3TrainStrategy

        collator = DataCollatorWithPacking() if packed else DataCollatorWithPadding()
        batch = TrainBatch(
            sample_ids=[str(i) for i in range(len(features))],
            strategy="eagle3",
            tensors=collator(features),
            metadata={"target_repr": "hidden_state"},
        )
        output = Eagle3TrainStrategy(model, target_head=head).forward_loss(batch)
        output.loss.backward()
        gradients = {
            name: parameter.grad.clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return output, gradients

    def test_production_loss_metrics_and_all_gradients_match_padding(self):
        cases = (
            (torch.float32, None),
            (torch.bfloat16, None),
            (torch.float32, {"rope_type": "dynamic", "factor": 2.0}),
        )
        for dtype, rope_scaling in cases:
            with self.subTest(dtype=dtype, rope_scaling=rope_scaling):
                model, head, features = self._fixtures(dtype, rope_scaling)
                packed_model = copy.deepcopy(model)
                padded, padded_grads = self._run(model, head, features, packed=False)
                packed, packed_grads = self._run(
                    packed_model, head, features, packed=True
                )
                if rope_scaling is not None:
                    # Initialization caches max_position_embeddings + 20 (180).
                    # The packed total is 201; packing must not extend that cache
                    # and change dynamic NTK scaling relative to the padded batch.
                    self.assertEqual(
                        packed_model.draft_model.midlayer.self_attn.rotary_emb.max_seq_len_cached,
                        180,
                    )
                tol = (
                    {"atol": 2e-6, "rtol": 2e-4}
                    if dtype == torch.float32
                    else {"atol": 3e-4, "rtol": 3e-2}
                )
                torch.testing.assert_close(packed.loss, padded.loss, **tol)
                for key in padded.metrics:
                    for expected, actual in zip(
                        padded.metrics[key], packed.metrics[key]
                    ):
                        torch.testing.assert_close(actual, expected, **tol)
                self.assertEqual(padded_grads.keys(), packed_grads.keys())
                for name in padded_grads:
                    torch.testing.assert_close(
                        packed_grads[name], padded_grads[name], msg=name, **tol
                    )

    def test_changing_previous_document_cannot_change_later_document_logits(self):
        model, head, features = self._fixtures()
        changed = copy.deepcopy(features)
        changed[0]["input_ids"].fill_(33)
        changed[0]["hidden_state"].mul_(100)
        changed[0]["target"].neg_()
        captured = []
        hook = model.draft_model.lm_head.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output.detach().clone())
        )
        try:
            self._run(model, head, features, packed=True)
            first = captured[:]
            captured.clear()
            model.zero_grad(set_to_none=True)
            self._run(model, head, changed, packed=True)
        finally:
            hook.remove()
        self.assertEqual(len(first), 4)
        self.assertEqual(len(captured), 4)
        start = features[0]["input_ids"].shape[1]
        for before, after in zip(first, captured):
            torch.testing.assert_close(
                before[:, start:], after[:, start:], atol=0, rtol=0
            )

    def test_fully_unsupervised_batch_has_zero_loss_and_finite_zero_gradients(self):
        model, head, features = self._fixtures()
        for feature in features:
            feature["loss_mask"].zero_()
        result, gradients = self._run(model, head, features, packed=True)
        self.assertEqual(float(result.loss.detach()), 0.0)
        for name, gradient in gradients.items():
            self.assertTrue(bool(torch.isfinite(gradient).all()), name)
            self.assertEqual(float(gradient.abs().max()), 0.0, name)

    def test_fsdp_forward_preserves_host_packing_metadata(self):
        import torch.distributed as dist
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if dist.is_initialized():
            self.skipTest("requires its own single-rank process group")
        model, head, features = self._fixtures()
        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group(
                "nccl",
                init_method=f"file://{directory}/process_group",
                rank=0,
                world_size=1,
            )
            try:
                wrapped = FSDP(model, use_orig_params=True)
                result, gradients = self._run(wrapped, head, features, packed=True)
                self.assertTrue(bool(torch.isfinite(result.loss)))
                self.assertTrue(gradients)
                for name, gradient in gradients.items():
                    self.assertTrue(bool(torch.isfinite(gradient).all()), name)
            finally:
                dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
