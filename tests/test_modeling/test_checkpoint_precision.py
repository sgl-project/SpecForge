"""Global initialization dtype must never round checkpoint-native parameters."""

import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.precision import (
    copy_parameter_preserving_dtype,
    load_state_dict_preserving_dtype,
    mixed_dtype_wrap_modules,
)
from tests.test_modeling.test_draft_precision import config


class MixedModel(nn.Module):
    """An architecture explicitly defining activation precision boundaries."""

    def __init__(self):
        super().__init__()
        self.low = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
        self.high = nn.Linear(8, 2, bias=False, dtype=torch.float32)
        self.observed_dtypes = None

    def forward(self, x):
        # Storage policy does not invent a model's compute precision policy.
        output = self.low(x.to(torch.bfloat16))
        result = self.high(output.float())
        self.observed_dtypes = (self.low.weight.dtype, self.high.weight.dtype)
        return result


def _distributed_worker(rank, rendezvous, sharding):
    import torch.distributed as dist
    from specforge.optimizer import BF16Optimizer
    from specforge.training.backend import FSDPTrainingBackend, ParallelConfig

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        torch.manual_seed(2)
        model = MixedModel().cuda(rank)
        backend = FSDPTrainingBackend(
            ParallelConfig(world_size=2, sharding_strategy=sharding),
            optimizer_factory=lambda m: BF16Optimizer(m, lr=0.01, total_steps=2),
        )
        wrapped = backend.prepare_model(model, optimizer_target=model)
        loss = wrapped(torch.randn(3, 8, device=rank)).square().mean()
        loss.backward()
        assert model.observed_dtypes == (torch.bfloat16, torch.float32)
        backend.optimizer.step()
        assert model.low.weight.dtype == torch.bfloat16
        assert model.high.weight.dtype == torch.float32
        assert torch.isfinite(loss)
    finally:
        dist.destroy_process_group()


class CheckpointPrecisionTest(unittest.TestCase):
    def test_state_loading_preserves_values_objects_and_missing_defaults(self):
        model = nn.ModuleDict({
            "low": nn.Linear(3, 2, bias=False),
            "high": nn.Linear(3, 2, bias=False),
            "new": nn.Linear(3, 2, bias=False),
        }).bfloat16()
        model["high"].requires_grad_(False)
        original = model["high"].weight
        source = {
            "low.weight": torch.randn(2, 3).bfloat16(),
            "high.weight": torch.full((2, 3), 1.000123, dtype=torch.float32),
        }
        result = load_state_dict_preserving_dtype(model, source, strict=False)
        self.assertEqual(result.missing_keys, ["new.weight"])
        self.assertIs(model["high"].weight, original)
        self.assertFalse(original.requires_grad)
        self.assertEqual(model["new"].weight.dtype, torch.bfloat16)
        for key, tensor in source.items():
            torch.testing.assert_close(model.state_dict()[key], tensor, rtol=0, atol=0)

    def test_tied_parameter_identity_is_preserved(self):
        model = nn.ModuleDict({"a": nn.Linear(2, 2), "b": nn.Linear(2, 2)}).bfloat16()
        model["b"].weight = model["a"].weight
        source = torch.full((2, 2), 1.000123)
        load_state_dict_preserving_dtype(
            model, {"a.weight": source, "b.weight": source}, strict=False
        )
        self.assertIs(model["a"].weight, model["b"].weight)
        torch.testing.assert_close(model["a"].weight, source, rtol=0, atol=0)

    def test_sharded_export_load_and_warm_start_keep_each_dtype(self):
        from specforge.training.model_loading import warm_start_draft_model

        source = AutoDraftModel.from_config(config(), torch_dtype=torch.bfloat16)
        source.fc.float()
        with torch.no_grad():
            source.fc.weight.fill_(1.000123)
        original = source.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            source.save_pretrained(directory, max_shard_size="30KB")
            self.assertTrue((Path(directory) / "model.safetensors.index.json").exists())
            self.assertEqual(json.loads((Path(directory) / "config.json").read_text())["dtype"], "bfloat16")
            restored = AutoDraftModel.from_pretrained(directory, dtype=torch.bfloat16)
            destination = AutoDraftModel.from_config(config(), torch_dtype=torch.bfloat16)
            warm_start_draft_model(
                destination, directory, draft_config=config(), strategy="dspark"
            )
            for model in (restored, destination):
                for key, tensor in original.items():
                    torch.testing.assert_close(model.state_dict()[key], tensor, rtol=0, atol=0)

    def test_missing_hf_weight_uses_default_without_recasting_existing_weights(self):
        from safetensors.torch import save_file

        source = AutoDraftModel.from_config(config(), torch_dtype=torch.bfloat16)
        source.fc.float()
        state = source.state_dict()
        state.pop("norm.weight")
        with tempfile.TemporaryDirectory() as directory:
            source.config.save_pretrained(directory)
            save_file(state, str(Path(directory) / "model.safetensors"))
            loaded = AutoDraftModel.from_pretrained(directory, dtype=torch.bfloat16)
            self.assertEqual(loaded.norm.weight.dtype, torch.bfloat16)
            torch.testing.assert_close(loaded.fc.weight, source.fc.weight, rtol=0, atol=0)

    def test_target_tables_keep_checkpoint_dtypes(self):
        from safetensors.torch import save_file
        from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead

        weights = {
            "model.embed_tokens.weight": torch.randn(4, 3).bfloat16(),
            "lm_head.weight": torch.full((4, 3), 1.000123),
        }
        with tempfile.TemporaryDirectory() as directory:
            save_file(weights, str(Path(directory) / "model.safetensors"))
            with patch(
                "specforge.modeling.target.target_utils.load_target_config",
                return_value=SimpleNamespace(vocab_size=4, hidden_size=3, pad_token_id=None),
            ):
                parts = TargetEmbeddingsAndHead.from_pretrained(
                    directory, device="cpu", dtype=torch.bfloat16
                )
            torch.testing.assert_close(parts.embed_tokens.weight, weights["model.embed_tokens.weight"], rtol=0, atol=0)
            torch.testing.assert_close(parts.lm_head.weight, weights["lm_head.weight"], rtol=0, atol=0)
            hidden = torch.randn(2, 3, dtype=torch.bfloat16, requires_grad=True)
            actual = parts.lm_head(hidden)
            expected = torch.nn.functional.linear(hidden.float(), weights["lm_head.weight"])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.sum().backward()
            self.assertIsNotNone(hidden.grad)

    def test_native_copy_does_not_round_through_destination(self):
        destination = nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
        source = torch.tensor([1.000123, 2.000456])
        copy_parameter_preserving_dtype(destination, source)
        torch.testing.assert_close(destination, source, rtol=0, atol=0)

    def test_fsdp_boundaries_are_only_added_for_mixed_models(self):
        model = MixedModel()
        self.assertEqual(mixed_dtype_wrap_modules(model), {model.low, model.high})
        self.assertEqual(mixed_dtype_wrap_modules(model.bfloat16()), set())
        model.high.bias = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "directly owned"):
            mixed_dtype_wrap_modules(model)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "two GPUs required")
    def test_two_rank_training_preserves_parameter_dtypes(self):
        for sharding in ("SHARD_GRAD_OP", "NO_SHARD"):
            with self.subTest(sharding=sharding), tempfile.TemporaryDirectory() as directory:
                torch.multiprocessing.spawn(
                    _distributed_worker,
                    args=(str(Path(directory) / "store"), sharding),
                    nprocs=2, join=True,
                )


if __name__ == "__main__":
    unittest.main()
