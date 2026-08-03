import copy
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from transformers import Qwen3Config

from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.muon import (
    ADAMW_OPTIMIZER,
    MUON_OPTIMIZER,
    capture_muon_parameter_metadata,
    partition_parameters_for_muon,
)
from specforge.optimizer import BF16Optimizer


class _TinyMarkovHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.markov_w1 = nn.Embedding(8, 3)
        self.markov_w2 = nn.Linear(3, 8, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(self.markov_w1(token_ids))


class _TinyDraft(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(6, 4, bias=False)
        self.layers = nn.ModuleList([nn.Linear(4, 4)])
        self.norm = nn.LayerNorm(4)
        self.embed_proj = nn.Sequential(nn.Linear(4, 4, bias=False))
        self.markov_head = _TinyMarkovHead()
        self.confidence_head = nn.Linear(4, 1)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.frozen_projection = nn.Linear(4, 4, bias=False)
        self.frozen_projection.requires_grad_(False)

    def forward(self, features: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(self.layers[0](self.fc(features)))
        return (
            self.lm_head(hidden).square().mean()
            + self.confidence_head(hidden).square().mean()
            + self.markov_head(token_ids).square().mean()
        )


class TestMuonParameterPartition(unittest.TestCase):
    def test_dspark_backbone_and_heads_are_partitioned_as_intended(self):
        config = Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=128,
            block_size=4,
            num_target_layers=4,
            dflash_config={
                "attention_mode": "gqa",
                "projector_type": "dspark",
                "target_layer_ids": [0],
                "markov_rank": 4,
                "markov_head_type": "vanilla",
                "enable_confidence_head": True,
                "confidence_head_with_markov": True,
            },
        )

        partition = partition_parameters_for_muon(DSparkDraftModel(config))

        muon_names = {item.name for item in partition.muon}
        adamw_names = {item.name for item in partition.adamw}
        self.assertIn("fc.weight", muon_names)
        self.assertIn("layers.0.self_attn.q_proj.weight", muon_names)
        self.assertIn("layers.0.mlp.down_proj.weight", muon_names)
        self.assertIn("markov_head.markov_w2.weight", adamw_names)
        self.assertIn("confidence_head.proj.weight", adamw_names)

    def test_only_hidden_linear_weights_use_muon(self):
        partition = partition_parameters_for_muon(_TinyDraft())

        self.assertEqual(
            tuple(item.name for item in partition.muon),
            ("fc.weight", "layers.0.weight"),
        )
        adamw_names = {item.name for item in partition.adamw}
        self.assertIn("layers.0.bias", adamw_names)
        self.assertIn("norm.weight", adamw_names)
        self.assertIn("embed_proj.0.weight", adamw_names)
        self.assertIn("markov_head.markov_w1.weight", adamw_names)
        self.assertIn("markov_head.markov_w2.weight", adamw_names)
        self.assertIn("confidence_head.weight", adamw_names)
        self.assertIn("lm_head.weight", adamw_names)
        self.assertNotIn("frozen_projection.weight", adamw_names)

    def test_pre_fsdp_metadata_preserves_logical_matrix_shape(self):
        model = _TinyDraft()
        metadata = capture_muon_parameter_metadata(model)
        logical_shape = model.fc.weight.shape
        model.fc.weight.data = model.fc.weight.data.reshape(-1)

        partition = partition_parameters_for_muon(model, metadata=metadata)

        fc_weight = next(item for item in partition.muon if item.name == "fc.weight")
        self.assertEqual(fc_weight.parameter.ndim, 1)
        self.assertEqual(fc_weight.logical_shape, logical_shape)

    def test_muon_rejects_a_model_without_hidden_matrices(self):
        model = nn.Sequential(nn.Embedding(8, 4), nn.LayerNorm(4))
        with self.assertRaisesRegex(ValueError, "no eligible hidden"):
            BF16Optimizer(model, lr=1e-3, optimizer_type=MUON_OPTIMIZER)


class TestBF16MuonOptimizer(unittest.TestCase):
    @staticmethod
    def _loss(model: nn.Module) -> torch.Tensor:
        features = torch.arange(12, dtype=torch.float32).reshape(2, 6) / 10
        token_ids = torch.tensor([1, 3])
        return model(features, token_ids)

    def test_hybrid_step_updates_both_groups_and_round_trips_state(self):
        torch.manual_seed(0)
        model = _TinyDraft()
        optimizer = BF16Optimizer(
            model,
            lr=1e-3,
            weight_decay=0.0,
            optimizer_type=MUON_OPTIMIZER,
            muon_lr=2e-3,
            muon_weight_decay=0.0,
            max_grad_norm=1.0,
            warmup_ratio=0.2,
            total_steps=10,
        )
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }

        self._loss(model).backward()
        grad_norm = optimizer.step()

        self.assertTrue(torch.isfinite(grad_norm))
        self.assertFalse(torch.equal(before["fc.weight"], model.fc.weight))
        self.assertFalse(torch.equal(before["lm_head.weight"], model.lm_head.weight))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertTrue(optimizer.optimizer.state)
        self.assertTrue(optimizer.aux_optimizer.state)

        saved_state = copy.deepcopy(optimizer.state_dict())
        restored_model = copy.deepcopy(model)
        restored = BF16Optimizer(
            restored_model,
            lr=1e-3,
            weight_decay=0.0,
            optimizer_type=MUON_OPTIMIZER,
            muon_lr=2e-3,
            muon_weight_decay=0.0,
            max_grad_norm=1.0,
            warmup_ratio=0.2,
            total_steps=10,
        )
        with patch("specforge.optimizer.print_on_rank0"):
            restored.load_state_dict(saved_state)

        self.assertEqual(restored.get_learning_rates(), optimizer.get_learning_rates())
        for restored_master, saved_master in zip(
            restored.fp32_params, saved_state["fp32_params"]
        ):
            torch.testing.assert_close(restored_master, saved_master)

        self._loss(model).backward()
        self._loss(restored_model).backward()
        optimizer.step()
        restored.step()
        for expected, actual in zip(model.parameters(), restored_model.parameters()):
            torch.testing.assert_close(actual, expected)

    def test_group_summary_is_complete_and_non_overlapping(self):
        model = _TinyDraft()
        optimizer = BF16Optimizer(
            model,
            lr=1e-3,
            optimizer_type=MUON_OPTIMIZER,
            muon_weight_decay=0.0,
            warmup_ratio=0.0,
            total_steps=10,
        )

        summary = optimizer.get_parameter_group_summary()
        muon_names = set(summary[MUON_OPTIMIZER]["names"])
        adamw_names = set(summary[ADAMW_OPTIMIZER]["names"])
        trainable_names = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertFalse(muon_names & adamw_names)
        self.assertEqual(muon_names | adamw_names, trainable_names)

    def test_muon_rejects_cpu_master_offload(self):
        with self.assertRaisesRegex(ValueError, "does not support optimizer CPU"):
            BF16Optimizer(
                _TinyDraft(),
                lr=1e-3,
                optimizer_type=MUON_OPTIMIZER,
                offload_master=True,
            )

    def test_adamw_checkpoint_schema_remains_backward_compatible(self):
        optimizer = BF16Optimizer(
            _TinyDraft(),
            lr=1e-3,
            optimizer_type=ADAMW_OPTIMIZER,
            warmup_ratio=0.0,
            total_steps=10,
        )
        state = optimizer.state_dict()

        self.assertEqual(
            set(state),
            {
                "optimizer_state_dict",
                "scheduler_state_dict",
                "max_grad_norm",
                "fp32_params",
            },
        )


def _run_sharded_muon_parity(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        full_parameter = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10
        full_gradient = torch.linspace(-0.5, 0.6, 12).reshape(4, 3)
        shard_sizes = (7, 5, 0)
        offset = sum(shard_sizes[:rank])
        shard_size = shard_sizes[rank]

        model = nn.Linear(3, 4, bias=False)
        model.weight.data.copy_(full_parameter)
        metadata = capture_muon_parameter_metadata(model)
        model.weight.data = full_parameter.reshape(-1)[
            offset : offset + shard_size
        ].clone()
        optimizer = BF16Optimizer(
            model,
            lr=2e-3,
            weight_decay=0.0,
            optimizer_type=MUON_OPTIMIZER,
            muon_weight_decay=0.1,
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_ns_steps=5,
            muon_adjust_lr_fn="match_rms_adamw",
            muon_metadata=metadata,
            max_grad_norm=1e9,
            warmup_ratio=0.0,
            total_steps=10,
        )
        optimizer.configure_grad_norm_reduction(enabled=True)

        expected_parameter = nn.Parameter(full_parameter.clone())
        expected_optimizer = torch.optim.Muon(
            [expected_parameter],
            lr=optimizer.get_learning_rate(),
            weight_decay=0.1,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            adjust_lr_fn="match_rms_adamw",
        )
        expected_parameter.grad = full_gradient.clone()
        expected_optimizer.step()

        if shard_size:
            model.weight.grad = full_gradient.reshape(-1)[
                offset : offset + shard_size
            ].clone()
        optimizer.step()

        expected_local = expected_parameter.detach().reshape(-1)[
            offset : offset + shard_size
        ]
        torch.testing.assert_close(model.weight, expected_local)
        expected_momentum = expected_optimizer.state[expected_parameter][
            "momentum_buffer"
        ].reshape(-1)[offset : offset + shard_size]
        torch.testing.assert_close(
            optimizer.optimizer.state[optimizer.fp32_params[0]]["momentum_buffer"],
            expected_momentum,
        )
    finally:
        dist.destroy_process_group()


class TestFSDPShardedMuon(unittest.TestCase):
    def test_sharded_update_matches_native_full_matrix_muon(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            init_file = os.path.join(temporary_directory, "process-group")
            mp.spawn(
                _run_sharded_muon_parity,
                args=(3, init_file),
                nprocs=3,
                join=True,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
