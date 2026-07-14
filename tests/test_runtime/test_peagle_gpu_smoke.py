# coding=utf-8
"""Single-GPU P-EAGLE train-step and checkpoint correctness gate."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()
MASK_TOKEN_ID = 31
PAD_TOKEN_ID = 0
SEQ_LEN = 8

TINY_PEAGLE_CONFIG = {
    "architectures": ["PEagleDraftModel"],
    "model_type": "llama",
    "hidden_act": "silu",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "num_hidden_layers": 2,
    "max_position_embeddings": 64,
    "vocab_size": 32,
    "draft_vocab_size": 32,
    "pad_token_id": PAD_TOKEN_ID,
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": False,
    "norm_before_residual": True,
}


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _snapshot(module: torch.nn.Module) -> list[torch.Tensor]:
    return [
        parameter.detach().clone()
        for parameter in module.parameters()
        if parameter.requires_grad
    ]


@unittest.skipUnless(CUDA, "P-EAGLE train-step gate requires CUDA")
class TestPEagleGpuSmoke(unittest.TestCase):
    def _assert_finite_nonzero(
        self,
        gradient: torch.Tensor | None,
        name: str,
    ) -> None:
        self.assertIsNotNone(gradient, f"{name} has no gradient")
        self.assertTrue(
            bool(torch.isfinite(gradient).all()),
            f"{name} has non-finite gradients",
        )
        self.assertGreater(
            gradient.float().abs().sum().item(),
            0.0,
            f"{name} gradient is all zero",
        )

    def _assert_module_gradients(self, module: torch.nn.Module, name: str) -> None:
        gradients = []
        for parameter_name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            self.assertIsNotNone(
                parameter.grad,
                f"{name}.{parameter_name} has no gradient",
            )
            self.assertTrue(
                bool(torch.isfinite(parameter.grad).all()),
                f"{name}.{parameter_name} has non-finite gradients",
            )
            gradients.append(parameter.grad)

        self.assertTrue(gradients, f"{name} has no trainable parameters")
        self.assertGreater(
            sum(gradient.float().abs().sum().item() for gradient in gradients),
            0.0,
            f"{name} gradients are all zero",
        )

    def _assert_module_updated(
        self,
        before: list[torch.Tensor],
        module: torch.nn.Module,
        name: str,
    ) -> None:
        after = [
            parameter for parameter in module.parameters() if parameter.requires_grad
        ]
        self.assertEqual(len(before), len(after))
        self.assertTrue(
            any(not torch.equal(old, new.detach()) for old, new in zip(before, after)),
            f"{name} was not updated",
        )

    def _capture_logits(
        self,
        wrapper: torch.nn.Module,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        captured_logits = []

        def capture_logits(_module, _inputs, output):
            captured_logits.append(output.detach().clone())

        handle = wrapper.draft_model.lm_head.register_forward_hook(capture_logits)
        was_training = wrapper.training
        try:
            wrapper.eval()
            with torch.no_grad():
                loss, _metrics = wrapper(**batch)
        finally:
            handle.remove()
            wrapper.train(was_training)

        self.assertEqual(len(captured_logits), 1)
        return loss.detach().clone(), captured_logits[0]

    def test_registry_train_step_and_checkpoint_parity(self):
        torch.cuda.set_device(0)
        self.addCleanup(torch.cuda.empty_cache)

        from specforge.core.peagle import OnlinePEagleModel
        from specforge.modeling.auto import AutoDraftModelConfig, AutoDraftModel
        from specforge.modeling.draft import PEagleDraftModel, resolve_draft
        from specforge.optimizer import BF16Optimizer

        with tempfile.TemporaryDirectory(prefix="peagle_gpu_") as workdir:
            config_path = os.path.join(workdir, "config.json")
            with open(config_path, "w", encoding="utf-8") as output:
                json.dump(TINY_PEAGLE_CONFIG, output)

            _seed(1234)
            config = AutoDraftModelConfig.from_file(config_path)
            draft_model = AutoDraftModel.from_config(
                config,
                torch_dtype=torch.bfloat16,
            ).cuda()
            self.assertIsInstance(draft_model, PEagleDraftModel)
            self.assertIs(resolve_draft("PEagleDraftModel"), PEagleDraftModel)
            self.assertTrue(draft_model.norm_before_residual)
            self.assertTrue(draft_model.layers[0].norm_before_residual)
            self.assertTrue(draft_model.embed_tokens.weight.requires_grad)
            self.assertEqual(draft_model.embed_tokens.padding_idx, PAD_TOKEN_ID)

            wrapper = OnlinePEagleModel(
                draft_model=draft_model,
                mask_token_id=MASK_TOKEN_ID,
                num_depths=4,
                down_sample_ratio=1.0,
                down_sample_ratio_min=1.0,
            ).cuda()
            wrapper.train()

            generator = torch.Generator().manual_seed(2027)
            hidden_states = torch.randn(
                1,
                SEQ_LEN,
                3 * TINY_PEAGLE_CONFIG["hidden_size"],
                generator=generator,
            ).to(device="cuda", dtype=torch.bfloat16)
            target = torch.randn(
                1,
                SEQ_LEN,
                TINY_PEAGLE_CONFIG["vocab_size"],
                generator=generator,
            ).to(device="cuda", dtype=torch.bfloat16)
            input_ids = torch.arange(
                1,
                SEQ_LEN + 1,
                device="cuda",
                dtype=torch.long,
            ).unsqueeze(0)

            batch = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "target": target,
                "loss_mask": torch.ones_like(input_ids),
                "hidden_states": hidden_states,
                "lengths": torch.tensor([SEQ_LEN]),
            }

            mask_hidden_before = draft_model.mask_hidden.detach().clone()
            mask_embedding_before = (
                draft_model.embed_tokens.weight[MASK_TOKEN_ID].detach().clone()
            )
            padding_embedding_before = (
                draft_model.embed_tokens.weight[PAD_TOKEN_ID].detach().clone()
            )
            layer_snapshots = [_snapshot(layer) for layer in draft_model.layers]

            optimizer = BF16Optimizer(
                wrapper,
                lr=1e-3,
                weight_decay=0.0,
                max_grad_norm=1.0,
                total_steps=4,
                warmup_ratio=0.0,
            )

            loss, metrics = wrapper(**batch)
            self.assertTrue(bool(torch.isfinite(loss)))
            self.assertGreater(metrics["full_acc_total"].item(), 0.0)
            loss.backward()
            torch.cuda.synchronize()

            self._assert_finite_nonzero(draft_model.mask_hidden.grad, "mask_hidden")
            embedding_gradient = draft_model.embed_tokens.weight.grad
            self._assert_finite_nonzero(
                embedding_gradient[MASK_TOKEN_ID],
                "mask-token embedding",
            )
            self.assertEqual(
                torch.count_nonzero(embedding_gradient[PAD_TOKEN_ID]).item(),
                0,
            )
            self._assert_module_gradients(draft_model.fc, "fc")
            self._assert_module_gradients(draft_model.norm, "norm")
            self._assert_module_gradients(draft_model.lm_head, "lm_head")
            for layer_index, layer in enumerate(draft_model.layers):
                self._assert_module_gradients(layer, f"layers.{layer_index}")

            grad_norm = optimizer.step()
            self.assertTrue(bool(torch.isfinite(grad_norm)))
            self.assertGreater(grad_norm.item(), 0.0)
            self.assertFalse(torch.equal(mask_hidden_before, draft_model.mask_hidden))
            self.assertFalse(
                torch.equal(
                    mask_embedding_before,
                    draft_model.embed_tokens.weight[MASK_TOKEN_ID],
                )
            )
            self.assertTrue(
                torch.equal(
                    padding_embedding_before,
                    draft_model.embed_tokens.weight[PAD_TOKEN_ID],
                )
            )
            for layer_index, (snapshot, layer) in enumerate(
                zip(layer_snapshots, draft_model.layers)
            ):
                self._assert_module_updated(
                    snapshot,
                    layer,
                    f"layers.{layer_index}",
                )

            reference_loss, reference_logits = self._capture_logits(wrapper, batch)
            self.assertTrue(bool(torch.isfinite(reference_loss)))
            self.assertTrue(bool(torch.isfinite(reference_logits).all()))
            expected_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in draft_model.state_dict().items()
            }

            checkpoint_dir = os.path.join(workdir, "checkpoint")
            draft_model.save_pretrained(checkpoint_dir)
            reloaded = AutoDraftModel.from_pretrained(
                checkpoint_dir,
                dtype=torch.bfloat16,
            ).cuda()
            self.assertIsInstance(reloaded, PEagleDraftModel)
            self.assertTrue(reloaded.norm_before_residual)
            self.assertTrue(reloaded.layers[0].norm_before_residual)

            actual_state = reloaded.state_dict()
            self.assertEqual(set(expected_state), set(actual_state))
            for name, expected in expected_state.items():
                with self.subTest(state=name):
                    torch.testing.assert_close(
                        actual_state[name].cpu(),
                        expected,
                        rtol=0,
                        atol=0,
                    )
            for name in ("cos_cached", "sin_cached"):
                expected_buffer = getattr(draft_model.rotary_emb, name)
                actual_buffer = getattr(reloaded.rotary_emb, name)
                self.assertTrue(
                    bool(torch.isfinite(expected_buffer).all()),
                    f"source rotary buffer {name} is non-finite",
                )
                self.assertTrue(
                    bool(torch.isfinite(actual_buffer).all()),
                    f"reloaded rotary buffer {name} is non-finite",
                )
                torch.testing.assert_close(
                    actual_buffer.float(),
                    expected_buffer.float(),
                    rtol=1e-3,
                    atol=1e-3,
                )

            reloaded_wrapper = OnlinePEagleModel(
                draft_model=reloaded,
                mask_token_id=MASK_TOKEN_ID,
                num_depths=4,
                down_sample_ratio=1.0,
                down_sample_ratio_min=1.0,
            ).cuda()
            torch.compiler.reset()
            actual_loss, actual_logits = self._capture_logits(
                reloaded_wrapper,
                batch,
            )
            self.assertTrue(bool(torch.isfinite(actual_logits).all()))
            torch.testing.assert_close(
                actual_logits,
                reference_logits,
                rtol=0,
                atol=0,
            )
            self.assertTrue(bool(torch.isfinite(actual_loss)))
            torch.testing.assert_close(
                actual_loss,
                reference_loss,
                rtol=0,
                atol=0,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
