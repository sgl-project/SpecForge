"""Fixed graph preparation and native GraphTrainer inheritance contracts."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torchtitan")

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.training.torchtitan.runtime import SpecForgeTitanTrainer


def _prepared_graph_window(loss_type):
    model = OnlineDFlashModel.__new__(OnlineDFlashModel)
    torch.nn.Module.__init__(model)
    model.block_size = 4
    model.num_anchors = 5
    model.loss_type = loss_type
    model.loss_decay_gamma = 2.0
    model.selector_loss_alpha = 0.75
    model.selector_warmup_ratio = 0.0
    model.selector_ramp_ratio = 0.5
    trainer = SpecForgeTitanTrainer.__new__(SpecForgeTitanTrainer)
    trainer._fixed_graph_shapes = True
    trainer.model_parts = [SimpleNamespace(training_model=model)]
    trainer.config = SimpleNamespace(
        model_spec=SimpleNamespace(model=SimpleNamespace(algorithm="dflash2")),
        schedule_total_steps=100,
        training=SimpleNamespace(steps=2),
    )
    mesh = SimpleNamespace(size=lambda: 1, get_local_rank=lambda: 0)
    trainer.parallel_dims = SimpleNamespace(
        get_optional_mesh=lambda *args, **kwargs: mesh, pp_enabled=False
    )
    trainer.metrics_processor = SimpleNamespace(
        should_log=lambda step: True,
        ntokens_since_last_log=0,
        data_loading_times=[],
    )
    trainer.gradient_accumulation_steps = 2
    trainer.num_pipeline_parallel_microbatches = 1
    trainer._anchor_generator = torch.Generator().manual_seed(1)
    trainer.device = torch.device("cpu")
    trainer.step = 1
    batches = []
    for length in (8, 5):
        mask = torch.zeros(1, 8, dtype=torch.long)
        mask[:, :length] = 1
        batches.append(({"input": mask, "loss_mask": mask}, mask))
    return trainer, model, trainer.batch_generator(batches)


@pytest.mark.parametrize("loss_type", ["dflash", "dpace"])
def test_graph_padding_preserves_objective_and_disables_cpu_diagnostics(loss_type):
    _trainer, model, batches = _prepared_graph_window(loss_type)
    prepared = [next(batches)[0], next(batches)[0]]
    expected = 0
    for inputs in prepared:
        assert inputs["anchor_positions"].shape == (1, 5)
        assert inputs["block_keep_mask"].shape == (1, 5)
        assert inputs["collect_detailed_metrics"] is False
        assert inputs["selector_loss_alpha"].shape == ()
        torch.testing.assert_close(inputs["selector_loss_alpha"], torch.tensor(0.015))
        if loss_type == "dpace":
            assert inputs["prepared_sequence_anchor_scale"].shape == (1, 5, 1)
        expected += model.prepared_objective_denominator(
            inputs["loss_mask"], inputs["anchor_positions"], inputs["block_keep_mask"]
        )
    torch.testing.assert_close(prepared[0]["objective_normalizer"], expected)
    assert prepared[0]["objective_normalizer"] is prepared[1]["objective_normalizer"]


def test_graph_trainer_reuses_native_forward_backward_and_specforge_preparation():
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

    from specforge.training.torchtitan.graph import SpecForgeGraphTrainer

    assert issubclass(SpecForgeGraphTrainer, GraphTrainer)
    assert issubclass(SpecForgeGraphTrainer, SpecForgeTitanTrainer)
    assert (
        SpecForgeGraphTrainer.batch_generator is SpecForgeTitanTrainer.batch_generator
    )
    assert SpecForgeGraphTrainer.Config().compile.mode == "aot_fx_trace"


def test_graph_replay_cannot_overwrite_accumulated_gradients(monkeypatch):
    from torchtitan.experiments.graph_trainer.common_utils import (
        accumulate_param_grads_,
    )
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

    from specforge.training.torchtitan.graph import SpecForgeGraphTrainer

    model = torch.nn.Linear(1, 1, bias=False)
    parameter = model.weight
    graph_owned_gradient = torch.zeros_like(parameter)

    def replay(self, *, value):
        graph_owned_gradient.fill_(value)
        accumulate_param_grads_([parameter], [graph_owned_gradient])
        return torch.tensor(value)

    monkeypatch.setattr(GraphTrainer, "forward_backward_step", replay)
    trainer = SpecForgeGraphTrainer.__new__(SpecForgeGraphTrainer)
    trainer.model_parts = [model]
    trainer.gradient_accumulation_steps = 3
    for value in (2.0, 3.0, 7.0):
        trainer.forward_backward_step(value=value)
    torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 12.0))


@pytest.mark.parametrize("initial_casts", [False, True])
@pytest.mark.parametrize("initial_division", [False, True])
@pytest.mark.parametrize("raise_inside", [False, True])
def test_compiler_numerics_restores_previous_policy(
    initial_casts, initial_division, raise_inside
):
    from torch._inductor import config

    from specforge.training.torchtitan.numerics import compiler_numerics

    with config.patch(
        {
            "emulate_precision_casts": initial_casts,
            "eager_numerics.division_rounding": initial_division,
        }
    ):
        outcome = (
            pytest.raises(RuntimeError, match="numerics restoration sentinel")
            if raise_inside
            else nullcontext()
        )
        with outcome:
            with compiler_numerics():
                assert config.emulate_precision_casts is True
                assert config.eager_numerics.division_rounding is True
                if raise_inside:
                    raise RuntimeError("numerics restoration sentinel")
        assert config.emulate_precision_casts is initial_casts
        assert config.eager_numerics.division_rounding is initial_division


def test_compiler_numerics_records_bf16_barriers_before_joint_trace():
    from torch._inductor import config
    from torch.fx.experimental.proxy_tensor import make_fx

    from specforge.training.torchtitan.numerics import compiler_numerics

    x = torch.tensor([1.0], dtype=torch.bfloat16, requires_grad=True)
    delta = torch.tensor([0.004], dtype=torch.bfloat16, requires_grad=True)

    def joint(x, delta):
        output = ((x + delta) - x) * x
        return output, *torch.autograd.grad(output.sum(), (x, delta))

    def barriers(graph):
        return [
            node
            for node in graph.graph.nodes
            if node.meta.get("low_precision_pointwise_barrier")
        ]

    with config.patch(emulate_precision_casts=False):
        unprotected = make_fx(joint, tracing_mode="fake")(x, delta)
        assert not barriers(unprotected)
        with compiler_numerics():
            protected = make_fx(joint, tracing_mode="fake")(x, delta)
        assert barriers(protected)
        # Applying the policy only after make_fx cannot restore these barriers.
        with compiler_numerics():
            assert not barriers(unprotected)
    expected = joint(x, delta)
    assert expected[0].item() == 0.0078125
    assert expected[1].item() == 0.0078125
    for actual, wanted in zip(protected(x, delta), expected):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires NVIDIA CUDA")
def test_compiler_numerics_preserves_bf16_conv_joint_forward_backward():
    from torch.fx.experimental.proxy_tensor import make_fx
    from torchtitan.experiments.graph_trainer.inductor_passes import (
        full_inductor_compilation_pass,
    )

    from specforge.training.torchtitan.graph import functionalize_fused_kernels
    from specforge.training.torchtitan.numerics import compiler_numerics
    from tests.test_modeling.test_dflash2_fused_conv import (
        _prepare_finish_step,
        _random_conv,
    )

    conv = _random_conv(256, 16, 2, 16, device="cuda", dtype=torch.bfloat16, seed=11)
    torch.manual_seed(72)
    inputs = torch.randn(1, 512, 256, device="cuda", dtype=torch.bfloat16)
    mixer = torch.randn_like(inputs)
    grad_seed = (torch.randn_like(inputs), torch.randn_like(inputs))
    expected = _prepare_finish_step(conv, inputs, mixer, grad_seed)
    inputs = inputs.detach().requires_grad_(True)
    mixer = mixer.detach().requires_grad_(True)
    conv.zero_grad(set_to_none=True)

    def joint(x, scale):
        prepared, dynamic = conv.prepare(x)
        finished = conv.finish(prepared * scale, dynamic)
        gradients = torch.autograd.grad(
            (prepared, finished),
            (x, scale, conv.base_kernel, conv.kernel_projection.weight),
            grad_seed,
        )
        return prepared, finished, *gradients

    with compiler_numerics():
        traced = make_fx(joint, tracing_mode="fake", _allow_non_fake_inputs=True)(
            inputs, mixer
        )
        transformed = functionalize_fused_kernels(traced, (inputs, mixer))
        assert any(
            "triton_kernel_wrapper_functional" in str(node.target)
            for node in transformed.graph.nodes
        )
        assert any(
            node.meta.get("low_precision_pointwise_barrier")
            for node in transformed.graph.nodes
        )
        transformed.graph.eliminate_dead_code()
        transformed.recompile()
        compiled = full_inductor_compilation_pass(transformed, (inputs, mixer))
        actual = compiled(inputs, mixer)
    for name, value in zip(expected, actual):
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0, msg=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires NVIDIA CUDA")
@pytest.mark.parametrize("kernel", ["conv", "head"])
def test_fused_kernel_joint_trace_survives_dead_code_elimination(kernel):
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch.fx.traceback import annotate_fn, preserve_node_meta

    from specforge.training.torchtitan.graph import functionalize_fused_kernels

    torch.manual_seed(27)
    if kernel == "conv":
        from specforge.modeling.draft.dflash2_conv_triton import (
            dflash2_grouped_conv_fused,
        )

        values = [
            torch.randn(
                2, 16, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
            ),
            torch.randn(
                2, 16, 2, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True
            ),
            torch.randn(2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True),
        ]

        def joint(*inputs):
            output = dflash2_grouped_conv_fused(*inputs, 4, 8)
            return output, *torch.autograd.grad(output.float().square().sum(), inputs)

    else:
        from specforge.core.dflash_head_triton import dflash_unary_head_fused

        values = [
            torch.randn(
                16, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
            ),
            torch.randn(128, 32, device="cuda", dtype=torch.bfloat16),
            torch.randint(128, (16,), device="cuda"),
        ]

        def joint(hidden, weight, targets):
            nll, top_values, top_ids, argmax_ids = dflash_unary_head_fused(
                hidden, weight, targets, 4
            )
            (gradient,) = torch.autograd.grad(
                nll.sum() + 0.3 * top_values.sum(), (hidden,)
            )
            return nll, top_values, top_ids, argmax_ids, gradient

    expected = joint(*values)
    with preserve_node_meta():
        traced = make_fx(
            annotate_fn({"module_fqn": "layers.0"})(joint), tracing_mode="fake"
        )(*values)
    assert any(
        "triton_kernel_wrapper_mutation" in str(n.target) for n in traced.graph.nodes
    )
    transformed = functionalize_fused_kernels(traced, values)
    assert all(
        node.meta.get("custom", {}).get("module_fqn") == "layers.0"
        for node in transformed.graph.nodes
        if "triton_kernel_wrapper_functional" in str(node.target)
    )
    transformed.graph.eliminate_dead_code()
    transformed.recompile()
    actual = transformed(*values)
    for found, wanted in zip(actual, expected):
        torch.testing.assert_close(found, wanted, rtol=0, atol=0)
