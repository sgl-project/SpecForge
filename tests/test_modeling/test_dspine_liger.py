import copy
import importlib.util
import unittest
from itertools import product
from unittest import mock

import torch

from specforge.algorithms.dspine import providers
from specforge.application import resolve_run
from specforge.config import Config
from specforge.modeling.draft import dflash_kernels
from specforge.runtime.contracts import TrainBatch
from specforge.training.model_loading import preflight_draft_kernels
from specforge.training.strategies.base import StepContext
from tests.draft_config_utils import draft_settings
from tests.test_modeling.test_dspine import tiny_config, training_inputs, training_model
from tests.test_modeling.test_dspine_accumulation import training_core

LIGER_AVAILABLE = importlib.util.find_spec("liger_kernel") is not None


def provider_config(enabled, attention_backend="sdpa"):
    return Config.model_validate(
        {
            "model": {
                "draft_model_config": draft_settings(),
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
    def setUp(self):
        platform = mock.patch.object(
            dflash_kernels, "current_liger_platform", return_value="cuda"
        )
        platform.start()
        self.addCleanup(platform.stop)

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

    def test_preflight_matches_explicit_opt_in(self):
        registration = providers.create_registration()
        for requested in (None, False, True):
            with (
                self.subTest(requested=requested),
                mock.patch(
                    "specforge.training.model_loading.resolve_draft_config",
                    return_value=tiny_config(),
                ),
                mock.patch.object(
                    dflash_kernels, "load_liger_dflash_kernels"
                ) as loader,
            ):
                cfg = resolve_run(provider_config(requested)).config
                preflight_draft_kernels(cfg, algorithm=registration)
                self.assertEqual(loader.call_count, int(requested is True))

    @unittest.skipIf(not LIGER_AVAILABLE, "requires optional liger-kernel")
    def test_provider_preserves_parameter_layout(self):
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
        optimized.load_state_dict(baseline.state_dict(), strict=True)

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
                    core, optimizer = training_core(model, 2)
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
