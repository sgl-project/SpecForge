"""MTP SDPA and chunked head loss preserve loss and gradients."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from specforge.algorithms.objectives import build_objective_model
from specforge.modeling import build_draft_model
from tests.test_algorithms.test_objectives import training_batch
from tests.test_checkpoint.test_native_presets import native_configs


def owned_mtp_config():
    return replace(
        next(native_configs(small=True)),
        target_bindings=(),
        target_logit_transform=None,
    )


def gradients(model):
    return {name: param.grad for name, param in model.named_parameters()}


class MTPBackendObjectiveTest(unittest.TestCase):
    def test_sdpa_chunked_loss_and_gradients(self):
        cases = [("cpu", torch.float32)]
        if torch.cuda.is_available():
            cases += [("cuda", torch.float32), ("cuda", torch.bfloat16)]
        for device, dtype in cases:
            rtol, atol = (3e-2, 3e-3) if dtype == torch.bfloat16 else (3e-4, 2e-6)
            config = owned_mtp_config()
            torch.manual_seed(31)
            reference = build_draft_model(config, device=device, dtype=dtype)
            batch = training_batch(config, device)
            batch = {
                k: v.to(dtype) if v.is_floating_point() else v for k, v in batch.items()
            }
            batch["position_ids"] = torch.arange(9, device=device)[None] + 17
            batch["attention_mask"][0, -2:] = 0
            expected = build_objective_model(
                reference, options={"objective_chunk_size": 0}
            )(batch)
            expected.loss.backward()
            for chunk_size in (0, 1, 3, 4096):
                with self.subTest(device=device, dtype=dtype, chunk_size=chunk_size):
                    model = build_draft_model(
                        config, device=device, dtype=dtype, backend="sdpa"
                    )
                    model.load_state_dict(reference.state_dict())
                    runtime = build_objective_model(
                        model, options={"objective_chunk_size": chunk_size}
                    )
                    with patch.object(
                        model.lm_head, "forward", wraps=model.lm_head.forward
                    ) as head:
                        actual = runtime(batch)
                        forward_calls = head.call_count
                        actual.loss.backward()
                        self.assertEqual(
                            head.call_count,
                            forward_calls * (2 if 0 < chunk_size < 8 else 1),
                        )
                        self.assertLessEqual(
                            max(call.args[0].shape[0] for call in head.call_args_list),
                            chunk_size or 8,
                        )
                    torch.testing.assert_close(
                        actual.loss, expected.loss, rtol=rtol, atol=atol
                    )
                    torch.testing.assert_close(
                        gradients(model), gradients(reference), rtol=rtol, atol=atol
                    )
                    torch.testing.assert_close(actual.ratios, expected.ratios)

    def test_frozen_backbone_and_empty_loss_mask(self):
        model = build_draft_model(owned_mtp_config()).requires_grad_(False)
        model.lm_head.requires_grad_(True)
        batch = training_batch(model.config.draft)
        runtime = build_objective_model(model, options={"objective_chunk_size": 2})
        runtime(batch).loss.backward()
        self.assertIsNotNone(model.lm_head.weight.grad)
        batch["loss_mask"].zero_()
        self.assertEqual(runtime(batch).loss.item(), 0)

    def test_chunk_sizes_validate(self):
        from specforge.algorithms.objectives.mtp import MTPObjective

        for value in (-1, True, 0.5):
            with self.assertRaises(ValueError):
                MTPObjective(objective_chunk_size=value)


if __name__ == "__main__":
    unittest.main()
