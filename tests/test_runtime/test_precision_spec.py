"""One precision per run: what each runtime resolves, and what the fsdp backend refuses."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
from torch import nn

from specforge.config import Config
from specforge.optimizer import BF16Optimizer, master_dtype_of
from specforge.training.assembly import _training_runtime
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.contracts import PrecisionSpec, TrainingRuntime

MODEL_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def _bf16_optimizer(module):
    return BF16Optimizer(module, lr=0.1, total_steps=1)


def _sgd(module):
    return torch.optim.SGD(module.parameters(), lr=0.1)


class PrecisionSpecTest(unittest.TestCase):
    def test_fsdp_stores_computes_and_reduces_in_the_model_dtype(self):
        for dtype in MODEL_DTYPES:
            with self.subTest(dtype=dtype):
                self.assertEqual(
                    PrecisionSpec.resolve(runtime="fsdp", model_dtype=dtype),
                    PrecisionSpec(
                        parameter_dtype=dtype,
                        compute_dtype=dtype,
                        gradient_dtype=dtype,
                        master_dtype=torch.float32,
                        artifact_dtype=dtype,
                    ),
                )

    def test_fsdp2_keeps_fp32_parameters_as_the_masters(self):
        for dtype in MODEL_DTYPES:
            with self.subTest(dtype=dtype):
                self.assertEqual(
                    PrecisionSpec.resolve(runtime="fsdp2", model_dtype=dtype),
                    PrecisionSpec(
                        parameter_dtype=torch.float32,
                        compute_dtype=dtype,
                        gradient_dtype=torch.float32,
                        master_dtype=None,
                        artifact_dtype=dtype,
                    ),
                )

    def test_unknown_runtime_and_integer_model_dtype_are_refused(self):
        with self.assertRaisesRegex(ValueError, "unknown runtime 'ddp'"):
            PrecisionSpec.resolve(runtime="ddp", model_dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "floating point"):
            PrecisionSpec.resolve(runtime="fsdp", model_dtype=torch.int64)

    def test_training_runtime_carries_the_precision_of_its_name(self):
        for name in ("fsdp", "fsdp2"):
            with self.subTest(name=name):
                runtime = TrainingRuntime.resolve(name, model_dtype=torch.bfloat16)
                self.assertEqual(runtime.name, name)
                self.assertEqual(
                    runtime.precision,
                    PrecisionSpec.resolve(runtime=name, model_dtype=torch.bfloat16),
                )


class BackendPrecisionTest(unittest.TestCase):
    def test_compute_or_gradients_in_another_dtype_are_refused(self):
        spec = PrecisionSpec.resolve(runtime="fsdp", model_dtype=torch.float32)
        for field in ("compute_dtype", "gradient_dtype"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "one dtype"):
                    FSDPTrainingBackend(
                        ParallelConfig(),
                        precision=replace(spec, **{field: torch.bfloat16}),
                    )
        # What fsdp2 resolves for a bfloat16 draft is not the fsdp runtime's.
        with self.assertRaisesRegex(ValueError, "one dtype"):
            FSDPTrainingBackend(
                ParallelConfig(),
                precision=PrecisionSpec.resolve(
                    runtime="fsdp2", model_dtype=torch.bfloat16
                ),
            )

    def test_artifact_in_another_dtype_is_refused(self):
        spec = PrecisionSpec.resolve(runtime="fsdp", model_dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "the dtype they are stored in"):
            FSDPTrainingBackend(
                ParallelConfig(), precision=replace(spec, artifact_dtype=torch.bfloat16)
            )

    def test_optimizer_keeping_other_masters_is_refused(self):
        spec = PrecisionSpec.resolve(runtime="fsdp", model_dtype=torch.float32)
        for factory, master_dtype in (
            (_sgd, torch.float32),
            (_bf16_optimizer, None),
        ):
            with self.subTest(factory=factory.__name__):
                backend = FSDPTrainingBackend(
                    ParallelConfig(),
                    optimizer_factory=factory,
                    precision=replace(spec, master_dtype=master_dtype),
                )
                with self.assertRaisesRegex(ValueError, "the run declares"):
                    backend.prepare_model(nn.Linear(2, 2), wrap=False)
                self.assertIsNone(backend.optimizer)
                with self.assertRaisesRegex(ValueError, "the run declares"):
                    backend.set_optimizer(factory(nn.Linear(2, 2)))
                self.assertIsNone(backend.optimizer)
        for factory, master_dtype in (
            (_sgd, None),
            (_bf16_optimizer, torch.float32),
        ):
            with self.subTest(factory=factory.__name__, master_dtype=master_dtype):
                backend = FSDPTrainingBackend(
                    ParallelConfig(),
                    optimizer_factory=factory,
                    precision=replace(spec, master_dtype=master_dtype),
                )
                backend.prepare_model(nn.Linear(2, 2), wrap=False)
                self.assertIsNotNone(backend.optimizer)

    def test_master_dtype_of_each_optimizer(self):
        model = nn.Linear(2, 2)
        self.assertEqual(master_dtype_of(_bf16_optimizer(model)), torch.float32)
        self.assertIsNone(master_dtype_of(_sgd(model)))
        with self.assertRaisesRegex(TypeError, "unknown optimizer object"):
            master_dtype_of(object())

    def test_stale_positional_parameter_dtype_is_refused(self):
        with self.assertRaises(TypeError):
            ParallelConfig(2, "NO_SHARD", torch.bfloat16)

    def test_trainable_parameters_in_another_dtype_are_refused(self):
        spec = PrecisionSpec.resolve(runtime="fsdp", model_dtype=torch.bfloat16)
        for strategy in ("NO_SHARD", "SHARD_GRAD_OP"):
            for wrap in (False, True):
                with self.subTest(strategy=strategy, wrap=wrap):
                    backend = FSDPTrainingBackend(
                        ParallelConfig(sharding_strategy=strategy), precision=spec
                    )
                    with self.assertRaisesRegex(ValueError, "declares torch.bfloat16"):
                        backend.prepare_model(nn.Linear(2, 2), wrap=wrap)
        # With an optimizer target, its parameters are the trainable draft.
        program = nn.Sequential(nn.Linear(2, 2).to(torch.bfloat16), nn.Linear(2, 2))
        backend = FSDPTrainingBackend(ParallelConfig(), precision=spec)
        with self.assertRaisesRegex(ValueError, "torch.float32"):
            backend.prepare_model(program, wrap=False, optimizer_target=program[1])

    def test_frozen_parameters_keep_the_dtype_they_were_built_in(self):
        table = nn.Embedding(4, 2).to(torch.bfloat16).requires_grad_(False)
        program = nn.Sequential(nn.Linear(2, 2), table)
        backend = FSDPTrainingBackend(
            ParallelConfig(),
            precision=PrecisionSpec.resolve(runtime="fsdp", model_dtype=torch.float32),
        )
        self.assertIs(backend.prepare_model(program, wrap=False), program)

    def test_sharding_wrapper_without_a_precision_is_refused(self):
        backend = FSDPTrainingBackend(ParallelConfig(sharding_strategy="SHARD_GRAD_OP"))
        with (
            patch("torch.distributed.fsdp.FullyShardedDataParallel") as wrapper,
            self.assertRaisesRegex(ValueError, "needs the run's PrecisionSpec"),
        ):
            backend.prepare_model(nn.Linear(2, 2))
        wrapper.assert_not_called()

    def test_unwrapped_and_replicated_programs_need_no_precision(self):
        model = nn.Linear(2, 2)
        self.assertIs(
            FSDPTrainingBackend(ParallelConfig()).prepare_model(model, wrap=False),
            model,
        )
        with patch(
            "torch.nn.parallel.DistributedDataParallel", return_value=model
        ) as wrapper:
            FSDPTrainingBackend(
                ParallelConfig(sharding_strategy="NO_SHARD")
            ).prepare_model(model)
        wrapper.assert_called_once()


class AssemblyPrecisionTest(unittest.TestCase):
    def test_sharding_wrapper_computes_and_reduces_in_the_configured_dtype(self):
        """Not in bfloat16, the dtype the wrapper used to fall back to."""
        for name, dtype in (("float16", torch.float16), ("float32", torch.float32)):
            with self.subTest(torch_dtype=name):
                cfg = Config.model_validate(
                    {
                        "model": {"target_model_path": "target", "torch_dtype": name},
                        "data": {"hidden_states_path": "features"},
                        "draft": "specforge/config/draft/presets/qwen3-8b-dflash.yaml",
                    }
                )
                runtime = _training_runtime(cfg)
                self.assertEqual(runtime.name, "fsdp")
                model = nn.Linear(2, 2, dtype=dtype)
                backend = FSDPTrainingBackend(
                    ParallelConfig(sharding_strategy="SHARD_GRAD_OP"),
                    precision=runtime.precision,
                )
                with patch(
                    "torch.distributed.fsdp.FullyShardedDataParallel",
                    return_value=model,
                ) as wrapper:
                    backend.prepare_model(model)
                precision = wrapper.call_args.kwargs["mixed_precision"]
                self.assertEqual(precision.param_dtype, dtype)
                self.assertEqual(precision.reduce_dtype, dtype)


if __name__ == "__main__":
    unittest.main()
