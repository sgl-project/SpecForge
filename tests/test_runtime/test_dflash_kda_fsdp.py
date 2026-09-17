# coding=utf-8
"""Two-rank FSDP smoke test for a real hybrid KDA decoder stack."""

import json
import math
import os
import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()
NGPU = torch.cuda.device_count() if CUDA else 0
WORLD_SIZE = 2


def _config():
    from transformers import Qwen3Config

    config = Qwen3Config(
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=3,
        head_dim=128,
        max_position_embeddings=128,
        vocab_size=64,
        tie_word_embeddings=False,
        attention_bias=False,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    config.architectures = ["DFlashDraftModel"]
    config.num_target_layers = 8
    config.block_size = 2
    config.draft_vocab_size = 64
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    config.use_sliding_window = False
    config.dflash_config = {
        "attention_modes": ["kda", "gqa", "kda"],
        "mask_token_id": 0,
    }
    config.linear_attn_config = {
        "head_dim": 128,
        "num_heads": 2,
        "short_conv_kernel_size": 4,
        "use_full_rank_gate": False,
        "gate_lower_bound": -5.0,
        "backend": "fla",
    }
    return config


def _worker(rank: int, world_size: int, port: int, results_dir: str) -> None:
    from tests.test_runtime import _fixtures as fixtures

    fixtures.init_rank_distributed(rank, world_size, port=str(port))
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from specforge.modeling.draft.dflash import DFlashDraftModel
        from specforge.training.backend import FSDPTrainingBackend, ParallelConfig

        torch.manual_seed(23 + rank)
        torch.cuda.manual_seed_all(23 + rank)
        model = DFlashDraftModel(_config()).to(
            device=torch.device("cuda", rank), dtype=torch.bfloat16
        )

        def optimizer_factory(module):
            return torch.optim.AdamW(module.parameters(), lr=1e-3)

        backend = FSDPTrainingBackend(
            ParallelConfig.from_distributed(param_dtype=torch.bfloat16),
            optimizer_factory=optimizer_factory,
        )
        wrapped = backend.prepare_model(model, wrap=True, optimizer_target=model)

        batch_size, draft_length, context_length = 2, 4, 5
        output = wrapped(
            position_ids=torch.arange(
                context_length + draft_length, device=torch.device("cuda", rank)
            ).expand(batch_size, -1),
            noise_embedding=torch.randn(
                batch_size,
                draft_length,
                model.config.hidden_size,
                device=torch.device("cuda", rank),
                dtype=torch.bfloat16,
            ),
            target_hidden=torch.randn(
                batch_size,
                context_length,
                len(model.target_layer_ids) * model.config.hidden_size,
                device=torch.device("cuda", rank),
                dtype=torch.bfloat16,
            ),
            attention_mask=torch.ones(
                batch_size,
                1,
                draft_length,
                context_length + draft_length,
                device=torch.device("cuda", rank),
                dtype=torch.bool,
            ),
        )
        loss = output.float().square().mean()
        backend.backward(loss)

        gradients = [
            parameter.grad
            for parameter in wrapped.parameters()
            if parameter.grad is not None and parameter.grad.numel()
        ]
        gradients_finite = all(torch.isfinite(gradient).all() for gradient in gradients)
        gradient_l1 = sum(gradient.float().abs().sum().item() for gradient in gradients)
        backend.optimizer.step()
        torch.cuda.synchronize()

        full_state = backend.state_dict()["model"]
        result = {
            "rank": rank,
            "loss": float(loss.item()),
            "gradient_l1": gradient_l1,
            "gradients_finite": bool(gradients_finite),
            "fsdp_units": sum(isinstance(module, FSDP) for module in wrapped.modules()),
            "wrapped_block_classes": sorted(
                block_class.__name__ for block_class in backend.auto_wrap_block_classes
            ),
            "gate_dtypes": None,
        }
        if full_state:
            result["gate_dtypes"] = {
                name: str(full_state[name].dtype)
                for name in (
                    "layers.0.self_attn.A_log",
                    "layers.0.self_attn.dt_bias",
                    "layers.2.self_attn.A_log",
                    "layers.2.self_attn.dt_bias",
                )
            }
        with open(os.path.join(results_dir, f"rank{rank}.json"), "w") as output_file:
            json.dump(result, output_file)
    finally:
        from specforge.distributed import destroy_distributed

        destroy_distributed()


@unittest.skipUnless(
    CUDA and NGPU >= WORLD_SIZE,
    "hybrid KDA FSDP training requires at least two CUDA devices",
)
class TestDFlashKDAFSDP(unittest.TestCase):
    def test_hybrid_decoder_blocks_train_under_fsdp(self):
        import torch.multiprocessing as mp

        from tests.utils import get_available_port

        with tempfile.TemporaryDirectory(prefix="dflash_kda_fsdp_") as workdir:
            mp.spawn(
                _worker,
                args=(WORLD_SIZE, get_available_port(), workdir),
                nprocs=WORLD_SIZE,
                join=True,
            )

            rank_results = []
            for rank in range(WORLD_SIZE):
                with open(os.path.join(workdir, f"rank{rank}.json")) as result_file:
                    rank_results.append(json.load(result_file))

        for result in rank_results:
            self.assertTrue(math.isfinite(result["loss"]), result)
            self.assertTrue(result["gradients_finite"], result)
            self.assertGreater(result["gradient_l1"], 0.0, result)
            self.assertGreaterEqual(result["fsdp_units"], 4, result)
            self.assertEqual(
                result["wrapped_block_classes"], ["Qwen3DFlashDecoderLayer"]
            )
        gathered_dtypes = next(
            result["gate_dtypes"]
            for result in rank_results
            if result["gate_dtypes"] is not None
        )
        self.assertEqual(set(gathered_dtypes.values()), {"torch.bfloat16"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
