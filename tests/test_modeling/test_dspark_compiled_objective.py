import copy
import tempfile
import unittest
from itertools import product
from pathlib import Path

import torch
import torch.nn.functional as functional

from specforge.modeling.draft.dspark import DSparkDraftModel
from tests.test_modeling.test_dspark_causal import build_model
from tests.test_modeling.test_dspark_target_kv import config, wrapper


class DSparkCompiledObjectiveTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cold_compile_inside_ddp_checkpoint(self):
        if torch.distributed.is_initialized():
            self.skipTest("requires an isolated process group")
        with tempfile.TemporaryDirectory() as directory:
            torch.distributed.init_process_group(
                "gloo",
                init_method=(Path(directory) / "rendezvous").as_uri(),
                rank=0,
                world_size=1,
            )
            try:
                model = build_model().cuda().bfloat16()
                model.flatten_projection = True
                model.cache_projection = True
                model.compile_objective = True
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0])
                tokens = torch.randint(0, 62, (2, 16), device="cuda")
                hidden = torch.randn(2, 16, 256, device="cuda").bfloat16()
                target = torch.randn(2, 16, 128, device="cuda").bfloat16()
                output = model(
                    tokens,
                    hidden,
                    torch.ones_like(tokens),
                    target_last_hidden_states=target,
                )
                output[0].backward()
                self.assertTrue(torch.isfinite(output[0]))
            finally:
                torch.distributed.destroy_process_group()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_full_model_loss_metrics_and_gradients(self):
        for conditioning, cache, chunk, autocast in product(
            ("hidden", "kv"), (False, True), (0, 1), (False, True)
        ):
            with self.subTest(
                conditioning=conditioning, cache=cache, chunk=chunk, autocast=autocast
            ):
                torch.manual_seed(42)
                baseline = (
                    (
                        build_model()
                        if conditioning == "hidden"
                        else wrapper(DSparkDraftModel(config()))
                    )
                    .cuda()
                    .bfloat16()
                )
                baseline.flatten_projection = True
                baseline.cache_projection = cache
                baseline.objective_chunk_blocks = chunk
                baseline.loss_decay_gamma = 7.0
                optimized = copy.deepcopy(baseline)
                optimized.compile_objective = True
                vocabulary = baseline.embed_tokens.num_embeddings
                width = baseline.embed_tokens.embedding_dim
                tokens = torch.randint(0, vocabulary - 1, (2, 16), device="cuda")
                mask = torch.ones_like(tokens)
                mask[:, :3] = 0
                mask[:, 10:12] = 0
                target = torch.randn(2, 32, width, device="cuda").bfloat16()[:, ::2]
                hidden = (
                    torch.randn(2, 16, 256, device="cuda").bfloat16()
                    if conditioning == "hidden"
                    else None
                )
                kwargs = {"target_last_hidden_states": target}
                if conditioning == "kv":
                    kwargs["target_kv"] = torch.randn(
                        2, 16, 2, 2, 2, 4, device="cuda"
                    ).bfloat16()
                outputs, gradients = [], []
                for model in (baseline, optimized):
                    torch.manual_seed(71)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                        output = model(tokens, hidden, mask, **kwargs)
                    output[0].backward()
                    outputs.append(output)
                    gradients.append(
                        {
                            name: parameter.grad.detach().float().clone()
                            for name, parameter in model.named_parameters()
                            if parameter.requires_grad and parameter.grad is not None
                        }
                    )
                torch.testing.assert_close(outputs[0][0], outputs[1][0])
                torch.testing.assert_close(outputs[0][1], outputs[1][1])
                for name, terms in outputs[0][2]["ratio_metrics"].items():
                    for expected, actual in zip(
                        terms, outputs[1][2]["ratio_metrics"][name]
                    ):
                        torch.testing.assert_close(
                            expected, actual, rtol=2e-5, atol=2e-5, msg=name
                        )
                self.assertEqual(gradients[0].keys(), gradients[1].keys())
                for name, expected in gradients[0].items():
                    torch.testing.assert_close(
                        expected, gradients[1][name], rtol=0.03, atol=0.002, msg=name
                    )
                expected = torch.cat(
                    [value.flatten() for value in gradients[0].values()]
                )
                actual = torch.cat([value.flatten() for value in gradients[1].values()])
                self.assertLess((actual - expected).norm() / expected.norm(), 0.01)
                self.assertGreater(
                    functional.cosine_similarity(expected, actual, dim=0), 0.9999
                )
                self.assertEqual(
                    baseline.state_dict().keys(), optimized.state_dict().keys()
                )


if __name__ == "__main__":
    unittest.main()
