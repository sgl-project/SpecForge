import copy
import importlib.util
import unittest
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import torch
import torch.nn.functional as F

from specforge.algorithms.dspine import providers
from specforge.config import Config
from specforge.modeling.draft import dflash_kernels
from specforge.modeling.draft.dspine import DSpineDraftModel
from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import StepContext
from tests.test_modeling.test_dspine import tiny_config, training_inputs, training_model
from tests.test_modeling.test_dspine_accumulation import training_core

LIGER_AVAILABLE = importlib.util.find_spec("liger_kernel") is not None


def provider_config(enabled, attention_backend="sdpa"):
    return Config.model_validate(
        {
            "model": {
                "target_model_path": "unused",
                "use_liger_kernel": enabled,
                "torch_dtype": "float32",
            },
            "data": {"hidden_states_path": "unused"},
            "training": {
                "strategy": "dspine",
                "total_steps": 600,
                "attention_backend": attention_backend,
            },
        }
    )


class DSpineLigerTest(unittest.TestCase):

    def test_disabled_provider_keeps_liger_lazy(self):
        registration = providers.create_registration()
        self.assertIs(registration.providers.model.build_draft, providers.build_draft)
        with (
            mock.patch.object(dflash_kernels, "load_liger_dflash_kernels") as loader,
            mock.patch(
                "specforge.algorithms.model_providers._device",
                return_value=torch.device("cpu"),
            ),
        ):
            draft = registration.providers.model.build_draft(
                provider_config(False), tiny_config(head_dim=16)
            )
        loader.assert_not_called()
        self.assertTrue(type(draft.norm).__module__.startswith("transformers"))
        self.assertTrue(type(draft.layers[0].mlp).__module__.startswith("transformers"))

    def test_enabled_provider_requires_optional_dependency(self):
        with mock.patch.object(
            dflash_kernels,
            "load_liger_dflash_kernels",
            side_effect=ImportError("missing optional liger"),
        ):
            with self.assertRaisesRegex(ImportError, "missing optional liger"):
                providers.build_draft(provider_config(True), tiny_config(head_dim=16))

    @unittest.skipIf(not LIGER_AVAILABLE, "requires optional liger-kernel")
    def test_provider_preserves_checkpoint_and_does_not_patch_globals(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            baseline = training_model(chunk_size=1, head_dim=16).draft_model
            with mock.patch(
                "specforge.algorithms.model_providers._device",
                return_value=torch.device("cpu"),
            ):
                optimized = providers.build_draft(
                    provider_config(True), tiny_config(head_dim=16)
                )
            self.assertTrue(type(optimized.norm).__module__.startswith("liger_kernel"))
            self.assertTrue(
                type(optimized.layers[0].mlp).__module__.startswith("liger_kernel")
            )
            self.assertEqual(set(optimized.state_dict()), set(baseline.state_dict()))
            optimized.load_state_dict(baseline.state_dict(), strict=True)
            optimized.save_pretrained(tmp_path)
            restored = DSpineDraftModel.from_pretrained(tmp_path)
            for name, value in baseline.state_dict().items():
                torch.testing.assert_close(
                    restored.state_dict()[name], value, atol=0, rtol=0
                )
            self.assertTrue(type(restored.norm).__module__.startswith("transformers"))

    @unittest.skipIf(
        not LIGER_AVAILABLE or not torch.cuda.is_available(),
        "requires CUDA and optional liger-kernel",
    )
    def test_liger_matches_loss_gradients_and_replayed_updates(self):
        for step, dtype, attention_backend in product(
            [100, 300], [torch.float32, torch.bfloat16], ["sdpa", "flex_attention"]
        ):
            with self.subTest(
                step=step, dtype=dtype, attention_backend=attention_backend
            ):
                baseline = training_model(chunk_size=1, head_dim=16)
                optimized = copy.deepcopy(baseline)
                optimized.draft_model = providers.build_draft(
                    provider_config(True, attention_backend), tiny_config(head_dim=16)
                )
                optimized.draft_model.load_state_dict(
                    baseline.draft_model.state_dict(), strict=True
                )
                inputs = training_inputs()
                for name in ("hidden_states", "target_last_hidden_states"):
                    inputs[name] = inputs[name].to(dtype)
                outputs = []
                gradients = []
                frozen_tables = []
                for model in (baseline, optimized):
                    model.attention_backend = attention_backend
                    model.draft_model.config._attn_implementation = attention_backend
                    model.to(device="cuda", dtype=dtype)
                    model.draft_model.transfer_codes = (
                        model.draft_model.transfer_codes.float()
                    )
                    frozen_tables.append(
                        model.draft_model.transfer_codes.detach().clone()
                    )
                    (core, optimizer) = training_core(model, 2)
                    torch.manual_seed(321)
                    for index in range(2):
                        output = core.train_step(
                            TrainBatch(
                                [str(index)],
                                "dspine",
                                {
                                    name: tensor[index : index + 1]
                                    for (name, tensor) in inputs.items()
                                },
                            ),
                            StepContext(global_step=step, total_steps=600),
                        )
                    self.assertTrue(output.optimizer_stepped)
                    self.assertEqual(core.accumulation_remainder, 0)
                    self.assertIs(model.lm_head.weight.grad, None)
                    self.assertIs(model.embed_tokens.weight.grad, None)
                    outputs.append(output.loss)
                    gradients.append(optimizer.gradients[-1])
                tolerance = 0.02 if dtype == torch.bfloat16 else 2e-05
                self.assertLessEqual(
                    abs(outputs[1] - outputs[0]),
                    max(tolerance, tolerance * abs(outputs[0])),
                )
                for actual, expected in zip(gradients[1], gradients[0]):
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(
                        actual, expected, rtol=tolerance, atol=tolerance
                    )
                expected = torch.cat(
                    [value.float().flatten() for value in gradients[0]]
                )
                actual = torch.cat([value.float().flatten() for value in gradients[1]])
                self.assertLess(
                    (actual - expected).norm() / expected.norm().clamp_min(1e-12),
                    tolerance,
                )
                self.assertGreater(F.cosine_similarity(actual, expected, dim=0), 0.999)
                for actual, expected in zip(
                    optimized.parameters(), baseline.parameters()
                ):
                    torch.testing.assert_close(
                        actual, expected, rtol=tolerance, atol=tolerance
                    )
                for model, original in zip((baseline, optimized), frozen_tables):
                    torch.testing.assert_close(
                        model.draft_model.transfer_codes, original, atol=0, rtol=0
                    )


if __name__ == "__main__":
    unittest.main()
