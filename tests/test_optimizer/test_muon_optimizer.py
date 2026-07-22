import copy
import functools
import os
import tempfile
import unittest
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from transformers import Qwen3Config

from specforge.config import Config
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.muon import (
    ADAMW_OPTIMIZER,
    MUON_OPTIMIZER,
    capture_muon_parameter_metadata,
    partition_parameters_for_muon,
)
from specforge.optimizer import BF16Optimizer
from specforge.training.assembly import _ConfiguredOptimizerFactory
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig


class _TinyMarkovHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.markov_w1 = nn.Embedding(8, 3)
        self.markov_w2 = nn.Linear(3, 8, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(self.markov_w1(token_ids))


class _TinyBlock(nn.Linear):
    pass


class _TinyDraft(nn.Module):
    _no_split_modules = ["_TinyBlock"]

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(6, 4, bias=False)
        self.layers = nn.ModuleList([_TinyBlock(4, 4)])
        self.norm = nn.LayerNorm(4)
        self.embed_proj = nn.Sequential(nn.Linear(4, 4, bias=False))
        self.markov_head = _TinyMarkovHead()
        self.confidence_head = nn.Linear(4, 1)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.frozen_projection = nn.Linear(4, 4, bias=False)
        self.frozen_projection.requires_grad_(False)

    def forward(self, features: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(self.layers[0](self.fc(features)))
        hidden = hidden + self.embed_proj(hidden)
        return (
            self.lm_head(hidden).square().mean()
            + self.confidence_head(hidden).square().mean()
            + self.markov_head(token_ids).square().mean()
        )


class TestMuonParameterPartition(unittest.TestCase):
    def test_shared_embedding_and_head_weights_stay_on_adamw(self):
        for excluded_module in (nn.Embedding(4, 4), nn.Linear(4, 4, bias=False)):
            with self.subTest(module=type(excluded_module).__name__):
                model = nn.Module()
                model.hidden = nn.Linear(4, 4, bias=False)
                model.lm_head = excluded_module
                model.lm_head.weight = model.hidden.weight
                partition = partition_parameters_for_muon(model)
                self.assertFalse(partition.muon)
                self.assertEqual(len(partition.adamw), 1)

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
    def test_resume_rejects_reordered_masters_and_changed_matrix_shapes(self):
        model = nn.Module()
        model.hidden = nn.Linear(4, 4, bias=False)
        model.lm_head = nn.Linear(4, 4, bias=False)
        optimizer = BF16Optimizer(model, lr=1e-3, optimizer_type="muon")
        state = optimizer.state_dict()

        reordered = nn.Module()
        reordered.lm_head = copy.deepcopy(model.lm_head)
        reordered.hidden = copy.deepcopy(model.hidden)
        restored = BF16Optimizer(reordered, lr=1e-3, optimizer_type="muon")
        with self.assertRaisesRegex(ValueError, "parameter layout"):
            restored.load_state_dict(state)

        for shape in ((4, 4), (2, 8)):
            model.hidden.weight.data = model.hidden.weight.data.reshape(shape)
            metadata = capture_muon_parameter_metadata(model)
            model.hidden.weight.data = model.hidden.weight.data.reshape(-1)
            sharded = BF16Optimizer(
                model, lr=1e-3, optimizer_type="muon", muon_metadata=metadata
            )
            if shape == (4, 4):
                state = sharded.state_dict()
            else:
                with self.assertRaisesRegex(ValueError, "parameter layout"):
                    sharded.load_state_dict(state)

    def test_both_schedulers_resume_and_reject_schedule_changes(self):
        for schedule in ("constant", "cosine"):
            with self.subTest(schedule=schedule):
                model = _TinyDraft()
                kwargs = dict(
                    lr=1e-3,
                    muon_lr=2e-3,
                    optimizer_type="muon",
                    lr_scheduler=schedule,
                    total_steps=8,
                    warmup_ratio=0.5,
                )
                optimizer = BF16Optimizer(model, **kwargs)
                for _ in range(2):
                    self._loss(model).backward()
                    optimizer.step()
                self.assertTrue(optimizer.optimizer.state)
                self.assertTrue(optimizer.aux_optimizer.state)
                state = copy.deepcopy(optimizer.state_dict())
                restored_model = copy.deepcopy(model)
                restored = BF16Optimizer(restored_model, **kwargs)
                with patch("specforge.optimizer.print_on_rank0"):
                    restored.load_state_dict(state)
                for _ in range(5):
                    self._loss(model).backward()
                    self._loss(restored_model).backward()
                    optimizer.step()
                    restored.step()
                    rates = optimizer.get_learning_rates()
                    self.assertEqual(restored.get_learning_rates(), rates)
                    self.assertAlmostEqual(rates["muon"], 2 * rates["adamw"])
                    if schedule == "constant":
                        self.assertEqual(rates["adamw"], 1e-3)
                    for expected, actual in zip(
                        model.parameters(), restored_model.parameters()
                    ):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                kwargs["lr_scheduler"] = (
                    "cosine" if schedule == "constant" else "constant"
                )
                with self.assertRaisesRegex(ValueError, "lr_scheduler"):
                    BF16Optimizer(model, **kwargs).load_state_dict(state)

    def test_invalid_step_leaves_both_optimizer_groups_unchanged(self):
        for invalid in ("gradient", "denominator"):
            with self.subTest(invalid=invalid):
                model = _TinyDraft()
                optimizer = BF16Optimizer(model, lr=1e-3, optimizer_type="muon")
                before = [master.detach().clone() for master in optimizer.fp32_params]
                scheduler_state = copy.deepcopy(optimizer.scheduler.state_dict())
                self._loss(model).backward()
                if invalid == "gradient":
                    model.fc.weight.grad.fill_(float("nan"))
                    error, kwargs = FloatingPointError, {}
                else:
                    error, kwargs = ValueError, {"loss_denominator": torch.tensor(0.0)}
                with self.assertRaises(error):
                    optimizer.step(**kwargs)
                self.assertFalse(optimizer.optimizer.state)
                self.assertFalse(optimizer.aux_optimizer.state)
                self.assertEqual(optimizer.scheduler.state_dict(), scheduler_state)
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                for expected, actual in zip(before, optimizer.fp32_params):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @staticmethod
    def _loss(model: nn.Module) -> torch.Tensor:
        features = torch.arange(12, dtype=torch.float32).reshape(2, 6) / 10
        token_ids = torch.tensor([1, 3])
        return model(features, token_ids)

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
                "lr_scheduler_type",
                "max_grad_norm",
                "fp32_params",
            },
        )


def _run_sharded_muon_parity(rank: int, world_size: int, init_file: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        process_group = dist.new_group([1, 2, 3])
        if rank == 0:
            return
        group_rank = dist.get_rank(process_group)
        for shape, shard_sizes, nesterov, adjustment in (
            ((4, 3), (7, 5, 0), True, "match_rms_adamw"),
            ((3, 4), (0, 12, 0), False, "original"),
        ):
            full_parameter = torch.arange(12, dtype=torch.float32).reshape(shape) / 10
            offset = sum(shard_sizes[:group_rank])
            shard_size = shard_sizes[group_rank]
            model = nn.Linear(shape[1], shape[0], bias=False)
            model.weight.data.copy_(full_parameter)
            metadata = capture_muon_parameter_metadata(model)
            model.weight.data = full_parameter.reshape(-1)[
                offset : offset + shard_size
            ].clone()
            optimizer = BF16Optimizer(
                model,
                lr=2e-3,
                optimizer_type="muon",
                muon_weight_decay=0.1,
                muon_nesterov=nesterov,
                muon_adjust_lr_fn=adjustment,
                muon_metadata=metadata,
                max_grad_norm=1e9,
                warmup_ratio=0.0,
                total_steps=10,
            )
            optimizer.configure_grad_norm_reduction(process_group=process_group)
            expected_parameter = nn.Parameter(full_parameter.clone())
            expected_optimizer = torch.optim.Muon(
                [expected_parameter],
                lr=2e-3,
                weight_decay=0.1,
                nesterov=nesterov,
                adjust_lr_fn=adjustment,
            )
            for step in range(4):
                full_gradient = torch.linspace(-0.5, 0.6, 12).reshape(shape) + step / 10
                expected_optimizer.param_groups[0]["lr"] = optimizer.get_learning_rate()
                expected_parameter.grad = full_gradient.clone() if step != 2 else None
                expected_optimizer.step()
                if shard_size and step != 2:
                    model.weight.grad = full_gradient.reshape(-1)[
                        offset : offset + shard_size
                    ].clone()
                optimizer.step()
                torch.testing.assert_close(
                    model.weight,
                    expected_parameter.detach().reshape(-1)[
                        offset : offset + shard_size
                    ],
                )
                torch.testing.assert_close(
                    optimizer.optimizer.state[optimizer.fp32_params[0]][
                        "momentum_buffer"
                    ],
                    expected_optimizer.state[expected_parameter][
                        "momentum_buffer"
                    ].reshape(-1)[offset : offset + shard_size],
                )
                if step == 1:
                    optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    finally:
        dist.destroy_process_group()


def _run_wrapped_muon_parity(rank: int, init_file: str, device_type: str) -> None:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    torch.set_num_threads(1)
    device = (
        torch.device(device_type, rank)
        if device_type == "cuda"
        else torch.device("cpu")
    )
    if device_type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl" if device_type == "cuda" else "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        for sharding in ("FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"):
            torch.manual_seed(10)
            model = _TinyDraft().to(device=device, dtype=torch.bfloat16)
            reference = copy.deepcopy(model)
            config = Config.model_validate(
                {
                    "model": {
                        "target_model_path": "target",
                        "draft_model_config": "draft.json",
                    },
                    "data": {"hidden_states_path": "/features"},
                    "training": {
                        "strategy": "dspark",
                        "optimizer": "muon",
                        "total_steps": 8,
                        "learning_rate": 1e-3,
                        "muon_learning_rate": 2e-3,
                        "lr_scheduler": "constant",
                        "warmup_ratio": 0.5,
                        "weight_decay": 0.01,
                        "muon_weight_decay": 0.1,
                    },
                }
            )
            backend = FSDPTrainingBackend(
                ParallelConfig(world_size=2, sharding_strategy=sharding),
                optimizer_factory=_ConfiguredOptimizerFactory(config),
            )
            # Explicit device_id lets real FSDP run on CPU in the unit suite.
            with patch(
                "torch.distributed.fsdp.FullyShardedDataParallel",
                functools.partial(FSDP, device_id=device),
            ):
                wrapped = backend.prepare_model(model, optimizer_target=model)
            reference_optimizer = _ConfiguredOptimizerFactory(config)(reference)
            reference_optimizer.configure_grad_norm_reduction(enabled=False)
            features = (
                torch.arange(12, device=device, dtype=torch.bfloat16).reshape(2, 6) / 10
            )
            token_ids = torch.tensor([1, 3], device=device)

            def train_step(step):
                backend.backward(wrapped(features + step / 10, token_ids))
                return backend.step(loss_denominator=torch.tensor(1.0, device=device))

            for step in range(3):
                norm = train_step(step)
                reference(features + step / 10, token_ids).backward()
                reference_norm = reference_optimizer.step()
                torch.testing.assert_close(norm, reference_norm, rtol=0.02, atol=1e-3)
                with backend._full_state_ctx():
                    actual = copy.deepcopy(
                        wrapped.state_dict()
                        if sharding != "NO_SHARD"
                        else model.state_dict()
                    )
                for name, expected in reference.state_dict().items():
                    torch.testing.assert_close(
                        actual[name], expected, rtol=0.02, atol=2e-3
                    )
                assert (
                    backend.optimizer.get_learning_rates()
                    == reference_optimizer.get_learning_rates()
                )
                if step == 0:
                    checkpoint = {
                        "model": actual,
                        "optimizer": copy.deepcopy(backend.optimizer.state_dict()),
                    }
            final_masters = [
                master.detach().clone() for master in backend.optimizer.fp32_params
            ]
            backend.load_state_dict(checkpoint)
            for step in (1, 2):
                train_step(step)
            for actual_master, expected_master in zip(
                backend.optimizer.fp32_params, final_masters
            ):
                torch.testing.assert_close(
                    actual_master, expected_master, rtol=0, atol=0
                )
    finally:
        dist.destroy_process_group()


class TestFSDPShardedMuon(unittest.TestCase):
    def test_sharded_update_matches_native_full_matrix_muon(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            init_file = os.path.join(temporary_directory, "process-group")
            mp.spawn(
                _run_sharded_muon_parity,
                args=(4, init_file),
                nprocs=4,
                join=True,
            )

    def test_real_fsdp_and_ddp_match_replicated_updates_and_resume(self):
        self._run_wrapped("cpu")

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
    def test_nccl_fsdp_and_ddp_match_replicated_updates_and_resume(self):
        self._run_wrapped("cuda")

    @staticmethod
    def _run_wrapped(device_type):
        with tempfile.TemporaryDirectory() as temporary_directory:
            mp.spawn(
                _run_wrapped_muon_parity,
                args=(os.path.join(temporary_directory, "process-group"), device_type),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
