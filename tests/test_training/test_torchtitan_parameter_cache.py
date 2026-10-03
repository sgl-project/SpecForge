"""Independent BF16 gradient oracle for repeated SimpleFSDP parameter reads."""

from contextlib import nullcontext

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint

pytest.importorskip("torchtitan")

from torchtitan.experiments.graph_trainer.simple_fsdp import (
    MixedPrecisionPolicy,
    data_parallel,
    disable_active_parametrization,
)

from specforge.training.torchtitan.parameter_cache import (
    graph_parameter_cache,
    install_graph_parameter_cache,
)


@pytest.fixture
def cpu_mesh(tmp_path):
    if torch.distributed.is_initialized():
        pytest.skip("Requires an isolated single-process Gloo group")
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1
    )
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        torch.distributed.destroy_process_group()


class RepeatedWeight(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.5]))

    def forward(self, coefficients, *, recompute=False):
        def chunk(coefficient):
            return (self.weight * coefficient).float().sum()

        terms = [
            checkpoint(chunk, value, use_reentrant=False) if recompute else chunk(value)
            for value in coefficients
        ]
        return sum(terms)


def make_model(mesh):
    return data_parallel(
        RepeatedWeight(),
        mesh,
        mode="replicate",
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        ),
    )


def coefficients():
    return torch.tensor([[1.0], [0.004]], dtype=torch.bfloat16, requires_grad=True)


@pytest.mark.parametrize("recompute", [False, True])
def test_cached_reads_match_independent_shared_bf16_gradient(cpu_mesh, recompute):
    inputs = coefficients()
    reference = torch.tensor([0.5], requires_grad=True)
    shared = reference.to(torch.bfloat16)
    reference_loss = sum((shared * value).float().sum() for value in inputs)
    reference_loss.backward()
    results = {}
    for cached in (False, True):
        model = make_model(cpu_mesh)
        leaf = model._parameters["weight"]
        install_graph_parameter_cache(model)
        with graph_parameter_cache(model) if cached else nullcontext():
            # Nested model-forward scopes must share the trainer's cache.
            with graph_parameter_cache(model) if cached else nullcontext():
                loss = model(inputs.detach().requires_grad_(), recompute=recompute)
            loss.backward()
        torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
        results[cached] = leaf.grad.to_local()
    torch.testing.assert_close(results[True], reference.grad, rtol=0, atol=0)
    assert results[False].item() == 1.003997802734375
    assert results[True].item() == 1.0078125


def test_cache_lifetime_nesting_and_exception_cleanup(cpu_mesh):
    model = make_model(cpu_mesh)
    install_graph_parameter_cache(model)
    assert model.weight is not model.weight
    with pytest.raises(RuntimeError, match="leave scope"):
        with graph_parameter_cache(model):
            first = model.weight
            with graph_parameter_cache(model):
                assert model.weight is first
            assert model.weight is first
            raise RuntimeError("leave scope")
    with torch.no_grad():
        model._parameters["weight"].add_(0.5)
    with graph_parameter_cache(model):
        assert model.weight is not first
        assert model.weight.item() == 1.0
    assert model.weight is not model.weight


def test_model_isolation_and_uninstalled_model_are_unchanged(cpu_mesh):
    left, right, uninstalled = [make_model(cpu_mesh) for _ in range(3)]
    install_graph_parameter_cache(left)
    install_graph_parameter_cache(right)
    with graph_parameter_cache(left):
        original = left.weight
        assert uninstalled.weight is not uninstalled.weight
        assert right.weight is not right.weight
        with graph_parameter_cache(right):
            assert right.weight is right.weight
            assert right.weight is not original
            assert left.weight is original
        assert right.weight is not right.weight
        assert left.weight is original


def test_state_dict_and_original_parameter_inspection(cpu_mesh):
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        set_model_state_dict,
    )

    model = make_model(cpu_mesh)
    leaf = model._parameters["weight"]
    before = get_model_state_dict(model)
    install_graph_parameter_cache(model)
    installed_type = type(model)
    install_graph_parameter_cache(model)
    assert type(model) is installed_type
    assert model._parameters["weight"] is leaf
    after = get_model_state_dict(model)
    assert before.keys() == after.keys() == {"weight"}
    torch.testing.assert_close(before["weight"], after["weight"], rtol=0, atol=0)
    with graph_parameter_cache(model):
        materialized = model.weight
        with disable_active_parametrization():
            assert model.weight is leaf
            assert isinstance(model.weight, DTensor)
        assert model.weight is materialized
    restored = make_model(cpu_mesh)
    install_graph_parameter_cache(restored)
    set_model_state_dict(restored, after)
    torch.testing.assert_close(restored.weight, model.weight, rtol=0, atol=0)


def test_functional_leaf_substitution_and_no_grad_do_not_poison_cache(cpu_mesh):
    model = make_model(cpu_mesh)
    install_graph_parameter_cache(model)
    with graph_parameter_cache(model):
        with torch.no_grad():
            inference_value = model.weight
        training_value = model.weight
        assert training_value is not inference_value
        assert training_value.requires_grad
        replacement = model._parameters["weight"].detach().clone().requires_grad_()
        with torch.no_grad():
            replacement.fill_(2.0)
        loss = torch.func.functional_call(
            model, {"weight": replacement}, (coefficients(),)
        )
        assert loss.item() == 2.00799560546875
        assert model.weight is training_value


def test_joint_fx_trace_preserves_shared_cast_backward(cpu_mesh):
    from torch.fx.experimental.proxy_tensor import make_fx

    model = make_model(cpu_mesh)
    install_graph_parameter_cache(model)
    leaf = model._parameters["weight"]
    inputs = coefficients()

    def joint(parameter, values):
        with graph_parameter_cache(model):
            loss = torch.func.functional_call(model, {"weight": parameter}, (values,))
            gradient = torch.autograd.grad(loss, parameter)[0].to_local()
        return loss, gradient

    traced = make_fx(joint)(leaf, inputs)
    expected = joint(leaf, inputs)
    actual = traced(leaf, inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual[1].item() == 1.0078125


def test_native_graph_tracer_reparameterization_and_replay(cpu_mesh):
    from torchtitan.experiments.graph_trainer.make_fx_tracer import (
        minimal_fx_tracer,
        run_traced,
    )

    model = make_model(cpu_mesh)
    install_graph_parameter_cache(model)
    inputs = coefficients()

    def joint(values):
        with graph_parameter_cache(model):
            loss = model(values, recompute=True)
            return loss, torch.autograd.grad(loss, tuple(model.parameters()))[0]

    with graph_parameter_cache(model):
        traced = minimal_fx_tracer(joint, module=model)(inputs)
    run = run_traced(traced, module=model, _validate_runtime=True)
    loss, gradient = run(inputs)
    assert loss.item() == 0.5019989013671875
    assert gradient.to_local().item() == 1.0078125
    with torch.no_grad():
        model._parameters["weight"].add_(0.5)
    changed_loss, _ = run(inputs)
    assert changed_loss.item() == 1.003997802734375
