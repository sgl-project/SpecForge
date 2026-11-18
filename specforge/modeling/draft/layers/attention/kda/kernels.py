"""KDA recurrence: a differentiable reference and the FLA chunked kernel.

Both take the raw decay gate and beta and apply Kimi's gate, beta sigmoid and
Q/K L2 normalization themselves. Shapes: ``q``/``k`` ``[N, T, H, K]``, ``v``
``[N, T, H, V]``, ``gate`` ``[N, T, H, K]``, ``beta`` ``[N, T, H]``; states are
FP32 ``[N, H, K, V]``. With ``cu_seqlens`` the batch is one flattened row of
variable-length sequences with one initial/final state each.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

FLA_REQUIREMENT = "fla-core==0.5.1"
# FLA processes chunks of 64 tokens; variable-length launches are padded to it.
FLA_CHUNK_SIZE = 64
_CUDA_MAX_GRID_DIM_Z = 65_535


def reference_kda(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    gate: Tensor,
    beta: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    lower_bound: float | None,
    *,
    initial_state: Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: Tensor | None = None,
    cu_seqlens_cpu: Tensor | None = None,
):
    """Step-by-step recurrence; the correctness oracle and the CPU path."""
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"variable-length KDA expects a flattened batch of 1, got {q.shape[0]}"
            )
        source = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
        bounds = [int(bound) for bound in source.tolist()]
        outputs, final_states = [], []
        for index, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            state = None if initial_state is None else initial_state[index : index + 1]
            output, final_state = reference_kda(
                q[:, start:end],
                k[:, start:end],
                v[:, start:end],
                gate[:, start:end],
                beta[:, start:end],
                A_log,
                dt_bias,
                lower_bound,
                initial_state=state,
                output_final_state=True,
            )
            outputs.append(output)
            final_states.append(final_state)
        output = torch.cat(outputs, dim=1)
        if output_final_state:
            return output, torch.cat(final_states, dim=0)
        return output

    q = F.normalize(q.float(), dim=-1).to(q.dtype)
    k = F.normalize(k.float(), dim=-1).to(k.dtype)
    beta = torch.sigmoid(beta.float()).to(q.dtype)
    gate_input = gate.float() + dt_bias.view(1, 1, *gate.shape[-2:])
    decay_scale = A_log.float().exp().view(1, 1, -1, 1)
    if lower_bound is None:
        log_decay = -decay_scale * F.softplus(gate_input)
    else:
        log_decay = float(lower_bound) * torch.sigmoid(decay_scale * gate_input)

    if initial_state is None:
        state = torch.zeros(
            q.shape[0],
            q.shape[2],
            q.shape[3],
            v.shape[3],
            dtype=torch.float32,
            device=q.device,
        )
    else:
        state = initial_state.to(device=q.device, dtype=torch.float32)
    outputs = []
    score_scale = q.shape[-1] ** -0.5
    for step in range(q.shape[1]):
        state = state * log_decay[:, step].exp().unsqueeze(-1)
        step_key = k[:, step].float()
        prediction = torch.einsum("bhd,bhdv->bhv", step_key, state)
        delta = (v[:, step].float() - prediction) * beta[:, step].float().unsqueeze(-1)
        state = state + torch.einsum("bhd,bhv->bhdv", step_key, delta)
        output = torch.einsum("bhd,bhdv->bhv", q[:, step].float(), state)
        outputs.append((output * score_scale).to(q.dtype))
    if outputs:
        output = torch.stack(outputs, dim=1)
    else:
        output = q.new_zeros((q.shape[0], 0, q.shape[2], v.shape[3]))
    if output_final_state:
        return output, state
    return output


def load_fla_chunk_kda() -> Callable:
    try:
        from fla.ops.kda import chunk_kda
    except ImportError as error:
        raise ImportError(
            f"KDA on CUDA requires {FLA_REQUIREMENT}; install SpecForge with the "
            "'kda' extra"
        ) from error
    return chunk_kda


def _pad_batch_to(tensor: Tensor, size: int) -> Tensor:
    padding = size - int(tensor.shape[0])
    if padding <= 0:
        return tensor
    return torch.cat((tensor, tensor.new_zeros((padding, *tensor.shape[1:]))), dim=0)


def _pad_time_to(tensor: Tensor, size: int) -> Tensor:
    padding = size - int(tensor.shape[1])
    if padding <= 0:
        return tensor
    return torch.cat(
        (tensor, tensor.new_zeros((tensor.shape[0], padding, *tensor.shape[2:]))),
        dim=1,
    )


def round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def _fla_kernel_kwargs(A_log: Tensor, dt_bias: Tensor, lower_bound) -> dict:
    return {
        # FSDP keeps one parameter dtype per decoder layer, so A_log/dt_bias are
        # stored in the model dtype (FP32 optimizer masters); the kernel reads FP32.
        "A_log": A_log.float(),
        "dt_bias": dt_bias.float(),
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "safe_gate": lower_bound is not None,
        "lower_bound": lower_bound,
    }


def _fla_kda_varlen(
    chunk_kda: Callable,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    gate: Tensor,
    beta: Tensor,
    *,
    initial_state: Tensor | None,
    output_final_state: bool,
    cu_seqlens: Tensor,
    kernel_kwargs: dict,
    cu_seqlens_cpu: Tensor | None = None,
):
    """One variable-length FLA launch with bounded launch shapes.

    FLA/TileLang specialize kernels on launch shapes, so the flattened batch is
    padded with one-token dummy sequences (the last absorbing the remainder) to
    a power-of-two sequence count and a chunk-aligned token count. Dummies never
    share state with real sequences and are sliced away.
    """
    if q.shape[0] != 1:
        raise ValueError(
            f"variable-length KDA expects a flattened batch of 1, got {q.shape[0]}"
        )
    bounds = cu_seqlens.to(device=q.device, dtype=torch.long)
    num_sequences = int(bounds.numel()) - 1
    total = int(q.shape[1])
    padded_sequences = 1 << max(num_sequences - 1, 0).bit_length()
    extra_sequences = padded_sequences - num_sequences
    padded_total = round_up(total + extra_sequences, FLA_CHUNK_SIZE)
    if padded_total > total and extra_sequences == 0:
        # The length remainder needs a dummy sequence to live in.
        padded_sequences *= 2
        extra_sequences = padded_sequences - num_sequences
        padded_total = round_up(total + extra_sequences, FLA_CHUNK_SIZE)
    bounds_cpu = cu_seqlens_cpu
    if extra_sequences:
        dummy_ends = total + torch.arange(
            1, extra_sequences + 1, device=bounds.device, dtype=torch.long
        )
        dummy_ends[-1] = padded_total
        bounds = torch.cat((bounds, dummy_ends))
        bounds_cpu = None  # padded on device only; FLA reads the device copy
    state = initial_state
    if state is not None and extra_sequences:
        state = _pad_batch_to(state, padded_sequences)
    output, final_state = chunk_kda(
        q=_pad_time_to(q, padded_total),
        k=_pad_time_to(k, padded_total),
        v=_pad_time_to(v, padded_total),
        g=_pad_time_to(gate, padded_total),
        beta=_pad_time_to(beta, padded_total),
        initial_state=state,
        output_final_state=output_final_state,
        cu_seqlens=bounds,
        **({"cu_seqlens_cpu": bounds_cpu} if bounds_cpu is not None else {}),
        **kernel_kwargs,
    )
    output = output[:, :total]
    if output_final_state:
        return output, final_state[:num_sequences]
    return output


def fla_kda(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    gate: Tensor,
    beta: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    lower_bound: float | None,
    *,
    initial_state: Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: Tensor | None = None,
    cu_seqlens_cpu: Tensor | None = None,
):
    """FLA ``chunk_kda`` within the CUDA grid-z launch limit; reference contract."""
    chunk_kda = load_fla_chunk_kda()
    kernel_kwargs = _fla_kernel_kwargs(A_log, dt_bias, lower_bound)
    if cu_seqlens is not None:
        return _fla_kda_varlen(
            chunk_kda,
            q,
            k,
            v,
            gate,
            beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            kernel_kwargs=kernel_kwargs,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )

    max_blocks_per_launch = max(1, _CUDA_MAX_GRID_DIM_Z // int(q.shape[2]))
    outputs, final_states = [], []
    for start in range(0, int(q.shape[0]), max_blocks_per_launch):
        end = min(start + max_blocks_per_launch, int(q.shape[0]))
        batch_size = end - start
        # TileLang specializes on the sequence count, which follows the sampled
        # anchor count; power-of-two buckets bound the compiled set.
        padded_size = min(1 << (batch_size - 1).bit_length(), max_blocks_per_launch)
        launch = {
            "q": _pad_batch_to(q[start:end], padded_size),
            "k": _pad_batch_to(k[start:end], padded_size),
            "v": _pad_batch_to(v[start:end], padded_size),
            "g": _pad_batch_to(gate[start:end], padded_size),
            "beta": _pad_batch_to(beta[start:end], padded_size),
        }
        if initial_state is not None:
            launch["initial_state"] = _pad_batch_to(
                initial_state[start:end], padded_size
            )
        output, final_state = chunk_kda(
            **launch, output_final_state=output_final_state, **kernel_kwargs
        )
        outputs.append(output[:batch_size])
        if output_final_state:
            final_states.append(final_state[:batch_size])
    output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
    if output_final_state:
        final_state = (
            final_states[0] if len(final_states) == 1 else torch.cat(final_states)
        )
        return output, final_state
    return output
