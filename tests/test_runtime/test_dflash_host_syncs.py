# coding=utf-8
"""DFlash-family micro-steps and optimizer boundaries stay off the host.

A synchronizing CUDA call blocks the Python thread until every queued kernel
drains, so the next micro-step cannot be enqueued behind the current backward.
These tests pin the synchronizations the trainer hot path is allowed:

* none in a micro-step whose integer features arrive on the host, including
  batches delivered by a pinned Mooncake receive pool to a CUDA consumer;
* exactly one per optimizer step (grad norm and loss denominator together).
"""

import unittest
import warnings
from unittest import mock

import torch
from torch import nn

from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import (
    DFlashTrainStrategy,
    DominoTrainStrategy,
    DSparkTrainStrategy,
    StepContext,
    StepOutput,
)

CUDA = torch.cuda.is_available()
_CTX = StepContext(global_step=5, total_steps=100, collect_detailed_metrics=False)


def _tiny_dflash2(attention_backend, *, sliding):
    from transformers import Qwen3Config

    from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
    from specforge.modeling.draft.dflash2 import DFlash2DraftModel

    config = Qwen3Config(
        architectures=["DFlash2DraftModel"],
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        num_target_layers=4,
        head_dim=16,
        max_position_embeddings=512,
        vocab_size=64,
        layer_types=["sliding_attention" if sliding else "full_attention"] * 2,
        sliding_window=32 if sliding else None,
        use_sliding_window=sliding,
        dflash_config={
            "block_size": 4,
            "conv_group_size": 4,
            "conv_kernel_size": 2,
            "mask_token_id": 63,
            "selector_rank": 4,
            "selector_top_k": 3,
            "target_layer_ids": [1, 2],
        },
    )
    config._attn_implementation = attention_backend
    device = torch.device("cuda")
    draft = DFlash2DraftModel(config).to(device, torch.bfloat16)
    return OnlineDFlashModel(
        draft_model=draft,
        target_lm_head=nn.Linear(64, 64, bias=False).to(device, torch.bfloat16),
        target_embed_tokens=nn.Embedding(64, 64).to(device, torch.bfloat16),
        mask_token_id=63,
        block_size=4,
        attention_backend=attention_backend,
        num_anchors=16,
        loss_decay_gamma=7.0,
        selector_loss_alpha=1.0,
    ).to(device)


def _features(seq_len=256, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return {
        "input_ids": torch.randint(0, 63, (1, seq_len), generator=generator),
        "loss_mask": (torch.rand(1, seq_len, generator=generator) > 0.2).long(),
        "hidden_states": torch.randn(1, seq_len, 128, generator=generator).to(
            torch.bfloat16
        ),
    }


def _batch(loss_mask_device):
    features = [_features(seed=0), _features(seed=1)]
    tensors = {
        name: torch.cat([feature[name] for feature in features]) for name in features[0]
    }
    tensors["hidden_states"] = tensors["hidden_states"].cuda()
    tensors["loss_mask"] = tensors["loss_mask"].to(loss_mask_device)
    return TrainBatch(
        sample_ids=["a", "b"], strategy="dflash", tensors=tensors, metadata={}
    )


def _micro_step(strategy, batch):
    strategy.forward_loss(batch, _CTX).loss.backward()


class _CountSyncs:
    """Count synchronizing CUDA calls (``set_sync_debug_mode('warn')``)."""

    def __enter__(self):
        torch.cuda.synchronize()
        self._warnings = warnings.catch_warnings(record=True)
        self.caught = self._warnings.__enter__()
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        return self

    def __exit__(self, *exc):
        torch.cuda.set_sync_debug_mode("default")
        self._warnings.__exit__(*exc)
        return False

    @property
    def count(self):
        return sum(
            "synchronizing CUDA operation" in str(w.message) for w in self.caught
        )


class _RecordingModel(nn.Module):
    """Captures what a DFlash-family strategy hands its model."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.kwargs = None

    def forward(self, **kwargs):
        self.kwargs = kwargs
        loss = self.weight * kwargs["hidden_states"].float().sum()
        return loss, torch.zeros(()), {}


class TestHostAnchorSizing(unittest.TestCase):
    """CPU: every DFlash-family strategy sizes anchors from a host loss mask."""

    def test_strategies_pass_host_anchor_width_with_device_hidden_states(self):
        device = "cuda" if CUDA else "cpu"
        loss_mask = torch.tensor([[1, 1, 0, 1, 1, 1], [1, 0, 1, 0, 1, 1]])
        tensors = {
            "input_ids": torch.zeros(2, 6, dtype=torch.long),
            "loss_mask": loss_mask,
            "hidden_states": torch.randn(2, 6, 4, device=device),
            "target_last_hidden_states": torch.randn(2, 6, 4, device=device),
        }
        for strategy_type in (
            DFlashTrainStrategy,
            DSparkTrainStrategy,
            DominoTrainStrategy,
        ):
            with self.subTest(strategy=strategy_type.name):
                model = _RecordingModel().to(device)
                strategy = strategy_type(model)
                strategy.forward_loss(
                    TrainBatch(
                        sample_ids=["a", "b"],
                        strategy=strategy.name,
                        tensors=tensors,
                        metadata={},
                    ),
                    _CTX,
                )
                # Row 0 has anchors at 0, 3, 4; row 1 only at 4.
                self.assertEqual(model.kwargs["max_valid_anchors"], 3)
                self.assertEqual(model.kwargs["loss_mask"].device.type, device)


@unittest.skipUnless(CUDA, "requires CUDA")
class TestDFlashMicroStepSyncs(unittest.TestCase):
    def test_micro_step_is_sync_free_with_host_loss_mask(self):
        for backend, sliding in (("sdpa", False), ("flex_attention", True)):
            with self.subTest(backend=backend, sliding=sliding):
                strategy = DFlashTrainStrategy(_tiny_dflash2(backend, sliding=sliding))
                for _ in range(2):  # compile / allocator warm-up
                    _micro_step(strategy, _batch("cpu"))
                batch = _batch("cpu")
                torch.cuda.synchronize()
                torch.cuda.set_sync_debug_mode("error")
                try:
                    _micro_step(strategy, batch)
                finally:
                    torch.cuda.set_sync_debug_mode("default")

    def test_device_loss_mask_still_falls_back_to_a_detected_sync(self):
        strategy = DFlashTrainStrategy(_tiny_dflash2("sdpa", sliding=False))
        _micro_step(strategy, _batch("cpu"))
        batch = _batch("cuda")
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            with self.assertRaisesRegex(RuntimeError, "synchronizing"):
                strategy.forward_loss(batch, _CTX)
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def test_pinned_mooncake_batches_feed_a_sync_free_micro_step(self):
        from specforge.algorithms.common.hidden_states_data import build_collator
        from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader
        from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
        from tests.test_runtime.test_mooncake_store import _FakeMooncakeStore, _meta

        store = MooncakeFeatureStore(
            store=_FakeMooncakeStore(), receive_buffers="pinned", retain_on_release=True
        )
        refs = [
            store.put(
                _features(seed=seed),
                sample_id=f"s{seed}",
                metadata={**_meta(), "strategy": "dflash"},
            )
            for seed in range(2)
        ]
        # The Trainer's loader for this store: device consumer, pinning
        # workers, and no redundant clone.
        loader = FeatureDataLoader(
            store,
            refs=refs,
            batch_size=2,
            collate_fn=build_collator(),
            device=store.consumer_device(),
            num_workers=1,
            pin_memory=True,
            strategy="dflash",
        )
        self.assertFalse(loader.clone_on_fetch)
        try:
            batch = next(iter(loader))
        finally:
            loader.close()
            store.close()

        tensors = batch.tensors
        self.assertTrue(tensors["hidden_states"].is_cuda)
        for name in ("input_ids", "loss_mask"):
            self.assertEqual(tensors[name].device.type, "cpu")
            self.assertTrue(tensors[name].is_pinned())
        expected = [_features(seed=seed) for seed in range(2)]
        for name in ("input_ids", "loss_mask", "hidden_states"):
            torch.testing.assert_close(
                tensors[name].cpu(),
                torch.cat([feature[name] for feature in expected]),
                rtol=0,
                atol=0,
            )

        strategy = DFlashTrainStrategy(_tiny_dflash2("sdpa", sliding=False))
        _micro_step(strategy, _batch("cpu"))
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            _micro_step(strategy, batch)
        finally:
            torch.cuda.set_sync_debug_mode("default")


class _WeightedStrategy:
    """Scalar-loss strategy with ``loss_terms`` and a host schedule metric."""

    name = "weighted"

    def __init__(self, model, denominator):
        self.model = model
        self.denominator = denominator

    def trainable_module(self):
        return self.model

    def forward_loss(self, batch, ctx=None):
        numerator = self.model(batch.tensors["x"]).float().square().sum()
        denominator = torch.full_like(numerator, self.denominator).detach()
        return StepOutput(
            loss=numerator / denominator,
            metrics={"lambda_base": 0.25},
            ratio_metrics={"acc": (numerator.detach(), denominator)},
            loss_terms=(numerator, denominator),
        )


def _boundary_core(*, denominator=4.0, accumulation_steps=2):
    from specforge.optimizer import BF16Optimizer
    from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
    from specforge.training.controller import TrainerCore

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 8)).to(
        "cuda", torch.bfloat16
    )
    backend = FSDPTrainingBackend(
        ParallelConfig(),
        optimizer_factory=lambda module: BF16Optimizer(
            module, lr=1e-3, max_grad_norm=0.05, total_steps=10, warmup_ratio=0.0
        ),
    )
    backend.prepare_model(model, wrap=False)
    strategy = _WeightedStrategy(model, denominator)
    return TrainerCore(strategy, backend, accumulation_steps=accumulation_steps)


def _x_batch():
    return TrainBatch(
        sample_ids=["s"],
        strategy="weighted",
        tensors={"x": torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)},
        metadata={},
    )


@unittest.skipUnless(CUDA, "requires CUDA")
class TestOptimizerBoundarySyncs(unittest.TestCase):
    def _boundary_syncs(self, core):
        for _ in range(2):  # warm up fused AdamW state and the allocator
            for _ in range(core.accumulation_steps):
                core.train_step(_x_batch())
        batches = [_x_batch() for _ in range(core.accumulation_steps)]
        with _CountSyncs() as micro:
            core.train_step(batches[0])
        with _CountSyncs() as boundary:
            result = core.train_step(batches[1])
        self.assertTrue(result.optimizer_stepped)
        return micro.count, boundary.count

    def test_optimizer_step_synchronizes_the_host_once(self):
        core = _boundary_core()
        self.assertTrue(core.backend.checks_loss_denominator)
        self.assertEqual(self._boundary_syncs(core), (0, 1))

    def test_dp_boundary_reduces_host_scalars_without_extra_syncs(self):
        core = _boundary_core()
        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch("torch.distributed.all_reduce"),
        ):
            self.assertEqual(self._boundary_syncs(core), (0, 1))

    def test_invalid_denominator_still_refuses_to_step(self):
        for denominator in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(denominator=denominator):
                core = _boundary_core(denominator=denominator, accumulation_steps=1)
                optimizer = core.backend.optimizer
                weights = [p.detach().clone() for p in optimizer.model_params]
                scheduler_epoch = optimizer.scheduler.last_epoch
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    core.train_step(_x_batch())
                self.assertFalse(optimizer.optimizer.state)
                self.assertEqual(optimizer.scheduler.last_epoch, scheduler_epoch)
                for before, param in zip(weights, optimizer.model_params):
                    torch.testing.assert_close(param, before, rtol=0, atol=0)
                    self.assertIsNone(param.grad)


if __name__ == "__main__":
    unittest.main()
