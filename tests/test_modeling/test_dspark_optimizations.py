import copy
import unittest
from itertools import product
from unittest import mock

import torch
import torch.nn.functional as functional

from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
from specforge.algorithms.dspark.providers import resume_contract
from specforge.algorithms.model_providers import build_dspark_model
from specforge.config import Config
from tests.test_modeling.test_dspark_causal import build_model


class DSparkOptimizationsTest(unittest.TestCase):

    def test_loss_metrics_and_all_gradients_match(self):
        for (dtype, device), chunk_size, (flatten, cache) in product(
            [(torch.float64, "cpu"), (torch.float32, "cuda"), (torch.bfloat16, "cuda")],
            [0, 2],
            [(True, False), (False, True), (True, True)],
        ):
            with self.subTest(
                dtype=dtype,
                device=device,
                chunk_size=chunk_size,
                flatten=flatten,
                cache=cache,
            ):
                if device == "cuda" and (not torch.cuda.is_available()):
                    self.skipTest("requires CUDA")
                torch.manual_seed(33)
                baseline = build_model().to(device=device, dtype=dtype)
                baseline.num_anchors = 5
                baseline.objective_chunk_blocks = chunk_size
                baseline.loss_decay_gamma = 7.0
                optimized = copy.deepcopy(baseline)
                optimized.flatten_projection = flatten
                optimized.cache_projection = cache
                tokens = torch.randint(0, 62, (2, 32), device=device)
                hidden = torch.randn(2, 32, 256, device=device, dtype=dtype)
                target = torch.randn(2, 64, 128, device=device, dtype=dtype)[:, ::2]
                mask = torch.ones_like(tokens)
                mask[:, :4] = 0
                mask[:, 22:25] = 0
                outputs = []
                gradients = []
                for model in (baseline, optimized):
                    torch.manual_seed(71)
                    output = model(
                        tokens, hidden, mask, target_last_hidden_states=target
                    )
                    output[0].backward()
                    outputs.append(output)
                    gradients.append(
                        {
                            name: parameter.grad.detach().float().clone()
                            for (name, parameter) in model.named_parameters()
                            if parameter.requires_grad
                        }
                    )
                tolerance = (
                    dict(rtol=0, atol=0)
                    if not flatten
                    else dict(
                        rtol=0.02 if dtype == torch.bfloat16 else 1e-05,
                        atol=0.002 if dtype == torch.bfloat16 else 1e-06,
                    )
                )
                torch.testing.assert_close(outputs[0][0], outputs[1][0], **tolerance)
                torch.testing.assert_close(outputs[0][1], outputs[1][1], **tolerance)
                for name, terms in outputs[0][2]["ratio_metrics"].items():
                    for expected, actual in zip(
                        terms, outputs[1][2]["ratio_metrics"][name]
                    ):
                        torch.testing.assert_close(
                            expected, actual, **tolerance, msg=name
                        )
                self.assertEqual(gradients[0].keys(), gradients[1].keys())
                for name, expected in gradients[0].items():
                    torch.testing.assert_close(
                        expected, gradients[1][name], **tolerance, msg=name
                    )
                expected = torch.cat(
                    [value.flatten() for value in gradients[0].values()]
                )
                actual = torch.cat([value.flatten() for value in gradients[1].values()])
                relative_error = (actual - expected).norm() / expected.norm()
                cosine = functional.cosine_similarity(expected, actual, dim=0)
                self.assertLess(
                    relative_error, 0.01 if dtype == torch.bfloat16 else 1e-05
                )
                self.assertGreater(cosine, 0.9999)
                self.assertEqual(
                    baseline.state_dict().keys(), optimized.state_dict().keys()
                )
                self.assertEqual(
                    resume_contract(None, baseline.draft_model, baseline),
                    resume_contract(None, optimized.draft_model, optimized),
                )

    def test_cache_reuses_only_frozen_projection_gemms(self):
        for trainable_head in [False, True]:
            with self.subTest(trainable_head=trainable_head):
                if not torch.cuda.is_available():
                    self.skipTest("requires CUDA kernel events")
                model = build_model(device="cuda")
                model.flatten_projection = True
                model.cache_projection = True
                model.lm_head.requires_grad_(trainable_head)
                tokens = torch.randint(0, 62, (2, 12), device="cuda")
                hidden = torch.randn(2, 12, 256, device="cuda")
                target = torch.randn(2, 12, 128, device="cuda")
                counts = []
                for enabled in (False, True):
                    model.cache_projection = enabled
                    model.zero_grad(set_to_none=True)
                    torch.manual_seed(71)
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        record_shapes=True,
                    ) as profiler:
                        (loss, _, _) = model(
                            tokens,
                            hidden,
                            torch.ones_like(tokens),
                            target_last_hidden_states=target,
                        )
                        loss.backward()
                        torch.cuda.synchronize()
                    counts.append(
                        sum(
                            (
                                len(event.kernels)
                                for event in profiler.events()
                                if event.name == "aten::mm"
                                and event.input_shapes == [[8, 128], [128, 64]]
                            )
                        )
                    )
                self.assertGreater(counts[0], 0)
                self.assertEqual(
                    counts[1], counts[0] if trainable_head else counts[0] // 2
                )

    def test_config_flags_are_dspark_only_and_reach_model(self):
        for option in ["dspark_flatten_projection", "dspark_cache_projection"]:
            with self.subTest(option=option):
                payload = {
                    "model": {"target_model_path": "unused"},
                    "data": {"hidden_states_path": "unused"},
                    "training": {"strategy": "dspark", option: True},
                }
                config = Config.model_validate(payload)
                reference = build_model()
                common = {
                    "draft_model": reference.draft_model,
                    "target_lm_head": reference.lm_head,
                    "target_embed_tokens": reference.embed_tokens,
                    "mask_token_id": 63,
                    "block_size": 4,
                }
                with mock.patch(
                    "specforge.algorithms.model_providers._build_dflash_family_model",
                    side_effect=lambda config, draft, tokenizer, factory: factory(
                        common
                    ),
                ):
                    model = build_dspark_model(
                        config, reference.draft_model, None, None, None
                    )
                self.assertTrue(isinstance(model, OnlineDSparkModel))
                self.assertIs(getattr(model, option.removeprefix("dspark_")), True)
                for strategy in ("dflash", "domino", "dspine", "eagle3"):
                    payload["training"]["strategy"] = strategy
                    with self.assertRaisesRegex(
                        ValueError, "require training.strategy=dspark"
                    ):
                        Config.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
