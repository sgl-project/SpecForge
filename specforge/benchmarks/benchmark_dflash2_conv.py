"""Benchmark the fused DFlash2 grouped convolution against the eager path.

Reports, for eager and fused (``SPECFORGE_DFLASH2_FUSED_CONV=0`` / unset):

* numerics of one convolution side against an FP64 evaluation of the same
  BF16 inputs and parameters;
* forward+backward time and peak memory of one decoder layer's four
  convolution calls, with and without the kernel projections;
* forward+backward time and peak memory of the full DFlash2 draft.

Example::

    python -m specforge.benchmarks.benchmark_dflash2_conv \\
        --config configs/qwen3.8-27b-dflash2.json --batch-size 2 \\
        --num-anchors 512 --context-length 4608
"""

import argparse
import contextlib
import copy
import os

import torch

from specforge.algorithms.common.dflash_family_model import (
    OnlineDFlashModel,
    create_dflash_block_mask,
)
from specforge.modeling.auto import AutoDraftModelConfig
from specforge.modeling.draft.dflash2 import (
    FUSED_CONV_ENV,
    DFlash2DraftModel,
    DFlashGroupedConv,
)
from specforge.modeling.draft.flex_attention_backend import flex_attention_backend


@contextlib.contextmanager
def conv_mode(mode):
    previous = os.environ.get(FUSED_CONV_ENV)
    os.environ[FUSED_CONV_ENV] = "1" if mode == "fused" else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(FUSED_CONV_ENV, None)
        else:
            os.environ[FUSED_CONV_ENV] = previous


def randomize_convs(model, generator_seed=0):
    """Give every convolution a non-trivial base and dynamic kernel."""
    generator = torch.Generator(device="cpu").manual_seed(generator_seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, DFlashGroupedConv):
                base = torch.randn(module.base_kernel.shape, generator=generator)
                base = base * 0.1
                base[:, 0] += 1.0
                module.base_kernel.copy_(base)
                weight = torch.randn(
                    module.kernel_projection.weight.shape, generator=generator
                )
                module.kernel_projection.weight.copy_(weight * 0.02)


def time_fwd_bwd(step, warmup, iters):
    """Return mean milliseconds and peak extra allocated GiB for ``step``."""
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        step()
    end.record()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - baseline) / 1024**3
    return start.elapsed_time(end) / iters, peak


def errors(actual, expected):
    diff = (actual.double() - expected.double()).abs()
    scale = expected.double().abs().max().clamp_min(1e-30)
    return (
        diff.max().item(),
        (diff.max() / scale).item(),
        diff.square().mean().sqrt().item(),
    )


def convolve_step(conv, inputs, delta, grad_output):
    inputs = inputs.detach().clone().requires_grad_(True)
    delta = delta.detach().clone().requires_grad_(True)
    conv.zero_grad(set_to_none=True)
    output = conv._convolve(inputs, delta, side=0)
    output.backward(grad_output)
    return {
        "output": output.detach(),
        "grad_x": inputs.grad,
        "grad_delta": delta.grad,
        "grad_base": conv.base_kernel.grad[0],
    }


def report_numerics(conv, batch_size, num_tokens):
    hidden = conv.base_kernel.shape[-1]
    shape = (batch_size, num_tokens, hidden)
    inputs = torch.randn(shape, device="cuda").bfloat16()
    delta = (
        torch.randn(batch_size, num_tokens, conv.taps, conv.num_groups, device="cuda")
        * 0.1
    ).bfloat16()
    grad_output = torch.randn(shape, device="cuda").bfloat16()

    with conv_mode("fused"):
        fused = convolve_step(conv, inputs, delta, grad_output)
        fused_fp32 = convolve_step(
            copy.deepcopy(conv).float(),
            inputs.float(),
            delta.float(),
            grad_output.float(),
        )
    with conv_mode("eager"):
        eager = convolve_step(conv, inputs, delta, grad_output)
        eager_fp32 = convolve_step(
            copy.deepcopy(conv).float(),
            inputs.float(),
            delta.float(),
            grad_output.float(),
        )
        reference = convolve_step(
            copy.deepcopy(conv).double(),
            inputs.double(),
            delta.double(),
            grad_output.double(),
        )

    print("\n=== Numerics: one convolution side, BF16 inputs vs FP64 ===")
    print("(max_abs, max_abs / max|ref|, rms)")
    columns = (
        "eager bf16 vs fp64",
        "fused bf16 vs fp64",
        "eager fp32 vs fp64",
        "fused fp32 vs fp64",
        "fused vs eager bf16",
    )
    print(f"{'tensor':<11} " + " ".join(f"{column:<34}" for column in columns))
    for name in reference:
        row = [
            errors(eager[name], reference[name]),
            errors(fused[name], reference[name]),
            errors(eager_fp32[name], reference[name]),
            errors(fused_fp32[name], reference[name]),
            errors(fused[name], eager[name]),
        ]
        cells = " ".join(f"{a:.2e} {r:.2e} {s:.2e}".ljust(34) for a, r, s in row)
        print(f"{name:<11} {cells}")


def benchmark_layer_convs(layer, batch_size, num_tokens, warmup, iters):
    """Time a decoder layer's four convolution calls in isolation."""
    hidden = layer.attention_conv.base_kernel.shape[-1]
    shape = (batch_size, num_tokens, hidden)
    tensors = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(4)
    ]
    grads = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    convs = (layer.attention_conv, layer.mlp_conv)
    deltas = [
        (
            torch.randn(
                batch_size,
                num_tokens,
                conv.taps,
                conv.num_groups,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.1
        ).requires_grad_(True)
        for conv in convs
        for _ in range(2)
    ]

    def with_projection():
        outputs = []
        for index, conv in enumerate(convs):
            prepared, kernel = conv.prepare(tensors[2 * index])
            outputs += [prepared, conv.finish(tensors[2 * index + 1], kernel)]
        torch.autograd.backward(outputs, grads)

    def convolution_only():
        outputs = []
        for index, conv in enumerate(convs):
            for side in range(2):
                call = 2 * index + side
                outputs.append(conv._convolve(tensors[call], deltas[call], side=side))
        torch.autograd.backward(outputs, grads)

    print(
        f"\n=== One decoder layer, 4 conv calls: B={batch_size}, "
        f"tokens={num_tokens}, hidden={hidden} ==="
    )
    print(f"{'variant':<34} {'eager ms':>9} {'fused ms':>9} {'speedup':>8}", end="")
    print(f" {'eager GiB':>10} {'fused GiB':>10}")
    for label, step in (
        ("_convolve x4 (fwd+bwd)", convolution_only),
        ("prepare/finish x2 incl. projection", with_projection),
    ):
        results = {}
        for mode in ("eager", "fused"):
            with conv_mode(mode):
                results[mode] = time_fwd_bwd(step, warmup, iters)
        (eager_ms, eager_gib), (fused_ms, fused_gib) = (
            results["eager"],
            results["fused"],
        )
        print(
            f"{label:<34} {eager_ms:>9.3f} {fused_ms:>9.3f} "
            f"{eager_ms / fused_ms:>7.2f}x {eager_gib:>10.3f} {fused_gib:>10.3f}"
        )


def benchmark_draft(model, config, args):
    """Time the full draft forward+backward on DFlash training-shaped inputs."""
    device = torch.device("cuda")
    batch_size, seq_len = args.batch_size, args.context_length
    block_size = model.block_size
    embed = torch.nn.Embedding(config.vocab_size, config.hidden_size).to(
        device=device, dtype=torch.bfloat16
    )
    embed.requires_grad_(False)
    online = OnlineDFlashModel(
        draft_model=model,
        target_lm_head=torch.nn.Identity(),
        target_embed_tokens=embed,
        mask_token_id=model.mask_token_id,
        block_size=block_size,
        attention_backend="flex_attention",
        num_anchors=args.num_anchors,
    )
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), device=device
    )
    loss_mask = torch.ones(batch_size, seq_len, device=device)
    target_hidden = torch.randn(
        batch_size,
        seq_len,
        len(model.target_layer_ids) * config.hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    torch.manual_seed(0)
    anchors, keep = online._sample_anchor_positions(seq_len, loss_mask, device)
    noise = online._create_noise_embed(input_ids, anchors, keep)
    position_ids = torch.cat(
        [
            torch.arange(seq_len, device=device).expand(batch_size, -1),
            online._create_position_ids(anchors),
        ],
        dim=1,
    )
    mask_args = dict(
        anchor_positions=anchors,
        block_keep_mask=keep,
        S=seq_len,
        block_size=block_size,
        device=device,
    )
    if flex_attention_backend() == "FLASH":
        mask_args["flex_block_size"] = (256, 128)
    attention_mask = create_dflash_block_mask(**mask_args)
    if model.sliding_window is not None:
        attention_mask = {
            "full_attention": attention_mask,
            "sliding_attention": create_dflash_block_mask(
                **mask_args, sliding_window=model.sliding_window
            ),
        }
    kernel_options = (
        {"BACKEND": "TRITON"}
        if torch.__version__ >= "2.11"
        else {"FORCE_USE_FLEX_ATTENTION": True}
    )
    grad_output = torch.randn(noise.shape, device=device, dtype=torch.bfloat16)

    def step():
        model.zero_grad(set_to_none=True)
        output = model(
            position_ids=position_ids,
            attention_mask=attention_mask,
            noise_embedding=noise,
            target_hidden=target_hidden,
            kernel_options=kernel_options,
        )
        output.backward(grad_output)

    print(
        f"\n=== Full DFlash2 draft ({config.num_hidden_layers} layers): "
        f"B={batch_size}, anchors={anchors.shape[1]}x{block_size}, "
        f"context={seq_len} ==="
    )
    results = {}
    for mode in ("eager", "fused"):
        with conv_mode(mode):
            results[mode] = time_fwd_bwd(step, args.warmup, args.iters)
            model.zero_grad(set_to_none=True)
    (eager_ms, eager_gib), (fused_ms, fused_gib) = results["eager"], results["fused"]
    print(f"{'':<34} {'eager':>9} {'fused':>9}")
    print(f"{'fwd+bwd ms':<34} {eager_ms:>9.2f} {fused_ms:>9.2f}")
    print(f"{'speedup':<34} {eager_ms / fused_ms:>8.3f}x")
    print(f"{'peak extra GiB':<34} {eager_gib:>9.3f} {fused_gib:>9.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/qwen3.8-27b-dflash2.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-anchors", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=4608)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--skip-draft", action="store_true")
    args = parser.parse_args()

    print("PyTorch:", torch.__version__, "GPU:", torch.cuda.get_device_name())
    config = AutoDraftModelConfig.from_file(args.config)
    config._attn_implementation = "flex_attention"
    torch.manual_seed(0)
    model = DFlash2DraftModel(config).to(device="cuda", dtype=torch.bfloat16)
    randomize_convs(model)
    num_tokens = args.num_anchors * model.block_size

    report_numerics(model.layers[0].attention_conv, args.batch_size, num_tokens)
    benchmark_layer_convs(
        model.layers[0], args.batch_size, num_tokens, args.warmup, args.iters
    )
    if not args.skip_draft:
        benchmark_draft(model, config, args)


if __name__ == "__main__":
    main()
