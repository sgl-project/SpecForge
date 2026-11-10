"""Validate a complete tensor assignment from metadata before any weight read."""

from dataclasses import dataclass

import torch

from .tensors import TensorRef


@dataclass(frozen=True)
class WeightPlan:
    weights: tuple[tuple[str, TensorRef], ...]
    bindings: tuple[tuple[str, TensorRef], ...] = ()
    allow_new: tuple[str, ...] = ()
    strict_copies: tuple[str, ...] = ()
    unused: tuple[str, ...] = ()


def _validate_dtype(name, source, destination, strict):
    if source == destination:
        return
    if not source.is_floating_point or not destination.is_floating_point:
        raise ValueError(f"{name}: dtype mismatch {source} -> {destination}")
    if strict:
        wider = {
            torch.float16: {torch.float32, torch.float64},
            torch.bfloat16: {torch.float32, torch.float64},
            torch.float32: {torch.float64},
        }
        if destination not in wider.get(source, set()):
            raise ValueError(
                f"{name}: copy would narrow/change precision {source} -> {destination}"
            )


def _validate_unquantized(name, ref):
    dtype = ref.info.dtype
    scales = {ref.key + "_scale", ref.key + "_scale_inv"}
    if (
        isinstance(dtype, str)
        or str(dtype).startswith(("torch.float8", "torch.float4"))
        or scales.intersection(ref.store.metadata)
    ):
        raise ValueError(
            f"{name} <- {ref.key}: unsupported quantized/scaled weight ({dtype}); "
            "provide unquantized weights"
        )


def make_weight_plan(
    model, weights, *, bindings=None, allow_new=(), strict_copies=(), unused=()
):
    """Require all persistent state, apart from explicitly new tensors and ties.

    The same function validates meta models and materialized models. Plans use
    exact tensor names; format adapters and initialization mappings expand their
    prefixes before calling it. No missing key is silently initialized.
    """
    state = model.state_dict(keep_vars=True)
    weights, bindings = dict(weights), dict(bindings or {})
    # Check selected sources before reporting their scale tensors as unexpected.
    for name, ref in (*weights.items(), *bindings.items()):
        _validate_unquantized(name, ref)
    allowed, strict = set(allow_new), set(strict_copies)
    if (weights.keys() | allowed) - state.keys():
        raise ValueError(
            f"unexpected draft tensors: {sorted((weights.keys() | allowed) - state.keys())}"
        )
    if weights.keys() & allowed:
        raise ValueError(
            f"loaded and allow_new overlap: {sorted(weights.keys() & allowed)}"
        )
    if strict - weights.keys():
        raise ValueError(
            f"strict copy has no source: {sorted(strict - weights.keys())}"
        )
    aliases = {}
    for name, value in state.items():
        aliases.setdefault(id(value), []).append(name)
    for names in aliases.values():
        sources = [weights[n] for n in names if n in weights]
        if sources:
            if any(ref != sources[0] for ref in sources[1:]) or allowed.intersection(
                names
            ):
                raise ValueError(f"conflicting assignments to tied tensors: {names}")
            for name in names:
                weights[name] = sources[0]
                if strict.intersection(names):
                    strict.add(name)
    missing = state.keys() - weights.keys() - allowed
    if missing:
        raise ValueError(f"missing draft tensors: {sorted(missing)}")
    expected_bindings = set(model.config.draft.target_bindings)
    if bindings.keys() != expected_bindings:
        raise ValueError(f"target bindings must be exactly {sorted(expected_bindings)}")
    for name, ref in list(weights.items()) + [
        (n + ".weight", r) for n, r in bindings.items()
    ]:
        destination = state[name] if name in state else model.get_parameter(name)
        if tuple(destination.shape) != ref.info.shape:
            raise ValueError(
                f"{name} <- {ref.key}: shape mismatch {ref.info.shape} != {tuple(destination.shape)}"
            )
        if name in strict and not ref.info.dtype.is_floating_point:
            raise ValueError(
                f"{name} <- {ref.key}: target copies require unquantized floating weights"
            )
        _validate_dtype(name, ref.info.dtype, destination.dtype, name in strict)
    return WeightPlan(
        tuple(sorted(weights.items())),
        tuple(sorted(bindings.items())),
        tuple(sorted(allowed)),
        tuple(sorted(strict)),
        tuple(sorted(unused)),
    )


def _local_tensor(tensor):
    from torch.distributed.tensor import DTensor, Partial

    if not isinstance(tensor, DTensor):
        return tensor, None
    if any(isinstance(p, Partial) for p in tensor.placements):
        raise ValueError("loading into Partial DTensor placements is not supported")
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    shape, offset = compute_local_shape_and_global_offset(
        tensor.shape, tensor.device_mesh, tensor.placements
    )
    return tensor.to_local(), tuple(
        slice(start, start + size) for start, size in zip(offset, shape)
    )


@torch.no_grad()
def apply_weight_plan(model, plan):
    """Copy into existing parameters, including rank-local DTensor slices.

    Revalidate against the actual destination before reading anything. Optimizer
    state, RNG restoration and post-load model reinitialization never occur here.
    """
    plan = make_weight_plan(
        model,
        dict(plan.weights),
        bindings=dict(plan.bindings),
        allow_new=plan.allow_new,
        strict_copies=plan.strict_copies,
        unused=plan.unused,
    )
    state = model.state_dict(keep_vars=True)
    destinations = {name: _local_tensor(value) for name, value in state.items()}
    bound = {
        name: _local_tensor(model.get_submodule(name).weight)
        for name, _ in plan.bindings
    }
    for name, (value, _) in {**destinations, **bound}.items():
        if value.is_meta:
            raise ValueError(f"{name}: materialize the model before applying weights")
    seen = set()
    for name, ref in plan.weights:
        if id(state[name]) not in seen:
            local, region = destinations[name]
            local.copy_(ref.read(region).to(device=local.device, dtype=local.dtype))
            seen.add(id(state[name]))
    for name, ref in plan.bindings:
        module = model.get_submodule(name)
        local, region = bound[name]
        value = ref.read(region).to(device=local.device, dtype=local.dtype)
        if region is not None:
            from torch.distributed.tensor import DTensor

            value = DTensor.from_local(
                value,
                module.weight.device_mesh,
                module.weight.placements,
                run_check=False,
                shape=module.weight.shape,
                stride=module.weight.stride(),
            )
        module.load_target_weight(value)
    return model
