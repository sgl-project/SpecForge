"""Domino cross-entropy with a portable reference and optional Triton path."""

import torch
import torch.nn.functional as F


def domino_weighted_cross_entropy(
    base_logits,
    correction_logits,
    targets,
    weights,
    block_size,
    suffix_start,
    *,
    use_fused: bool,
):
    """Return weighted final/base loss sums and predictions for Domino logits.

    ``use_fused=True`` requires CUDA and imports the Triton implementation
    lazily. Otherwise, the reference path materializes final logits without
    requiring Triton. Only the two logit tensors are differentiable; targets,
    weights, and layout arguments are treated as fixed inputs.

    Tensor layout:
        ``num_rows = num_blocks * block_size``
        ``base_logits``: ``[num_rows, vocab_size]``
        ``correction_logits``: ``[num_blocks * suffix_size, vocab_size]``
        ``targets``: ``[num_rows]`` integer indices in ``[0, vocab_size)``
        ``weights``: ``[num_rows]``

    Returns:
        ``final_loss_sum``: weighted corrected-logit losses summed over rows.
        ``base_loss_sum``: weighted base-logit losses summed over rows.
        ``final_pred``: ``[num_rows]`` predictions from corrected logits.
        ``base_pred``: ``[num_rows]`` predictions from base logits.
    """
    if base_logits.dim() != 2 or correction_logits.dim() != 2:
        raise ValueError("Domino cross entropy expects 2D logits")
    if block_size < 1 or not 0 <= suffix_start < block_size:
        raise ValueError("suffix_start must select a non-empty Domino suffix")
    num_rows, vocab_size = base_logits.shape
    if vocab_size < 1:
        raise ValueError("Domino cross entropy requires a non-empty vocabulary")
    if num_rows % block_size:
        raise ValueError("base-logit rows must be divisible by block_size")
    num_blocks = num_rows // block_size
    suffix_size = block_size - suffix_start
    expected_correction_rows = num_blocks * suffix_size
    if correction_logits.shape != (expected_correction_rows, vocab_size):
        raise ValueError("correction logits do not match the configured suffix width")
    if targets.shape != (num_rows,) or weights.shape != (num_rows,):
        raise ValueError("targets and weights must have one value per logit row")
    if targets.dtype != torch.long:
        raise ValueError("targets must contain torch.long class indices")
    if base_logits.device != correction_logits.device or any(
        tensor.device != base_logits.device for tensor in (targets, weights)
    ):
        raise ValueError("Domino cross-entropy inputs must be on the same device")
    if base_logits.dtype != correction_logits.dtype:
        raise ValueError("base and correction logits must have the same dtype")
    if not base_logits.is_floating_point():
        raise ValueError("base and correction logits must be floating point")

    if not use_fused:
        return _domino_weighted_cross_entropy_reference(
            base_logits,
            correction_logits,
            targets,
            weights,
            block_size,
            suffix_start,
        )
    if not base_logits.is_cuda:
        raise ValueError("Fused Domino cross entropy requires CUDA logits")
    if base_logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            "Fused Domino cross entropy requires FP16, BF16, or FP32 logits"
        )

    try:
        from specforge.core.domino_loss_triton import (
            domino_weighted_cross_entropy_fused,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "triton" or exc.name.startswith("triton."):
            raise ImportError(
                "Fused Domino cross entropy requires Triton; install Triton or "
                "call with use_fused=False."
            ) from exc
        raise

    return domino_weighted_cross_entropy_fused(
        base_logits,
        correction_logits,
        targets,
        weights,
        block_size,
        suffix_start,
    )


def _domino_weighted_cross_entropy_reference(
    base_logits,
    correction_logits,
    targets,
    weights,
    block_size,
    suffix_start,
):
    """Reference implementation that materializes the corrected logits."""
    num_blocks = base_logits.shape[0] // block_size
    suffix_size = block_size - suffix_start
    base_logits_3d = base_logits.reshape(num_blocks, block_size, -1)
    correction_logits_3d = correction_logits.reshape(num_blocks, suffix_size, -1)
    final_logits = torch.cat(
        [
            base_logits_3d[:, :suffix_start],
            base_logits_3d[:, suffix_start:] + correction_logits_3d,
        ],
        dim=1,
    ).reshape_as(base_logits)
    loss_logits = (
        final_logits.float()
        if final_logits.dtype in (torch.float16, torch.bfloat16)
        else final_logits
    )
    base_loss_logits = (
        base_logits.float()
        if base_logits.dtype in (torch.float16, torch.bfloat16)
        else base_logits
    )
    final_losses = F.cross_entropy(loss_logits, targets, reduction="none")
    base_losses = F.cross_entropy(base_loss_logits, targets, reduction="none")
    final_loss_sum = (final_losses * weights).sum()
    base_loss_sum = (base_losses * weights).sum()
    return (
        final_loss_sum,
        base_loss_sum,
        final_logits.argmax(-1),
        base_logits.argmax(-1),
    )


__all__ = ["domino_weighted_cross_entropy"]
