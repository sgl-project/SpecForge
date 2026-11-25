"""Teacher-forced MMFlash scores for aligned, single-image responses.

The KL is KL(with image || without image), in nats. Features for the DFlash
draft come only from the with-image pass. The distribution for token t is
read at t-1, including the first supervised response token.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack

import torch

Batch = Mapping[str, torch.Tensor]


def _capture(target, inputs: Batch, keep: int, layers: Sequence[int]):
    # HF replaces the last returned hidden state with the normalized LM input.
    # DFlash needs decoder block outputs before the final norm. Capture only
    # the requested Qwen language layers, in the configured order.
    captured = {}

    def record(module, _args, output):
        captured[module] = output[0] if isinstance(output, tuple) else output

    blocks = []
    with ExitStack() as hooks:
        if layers:
            decoder = target.model.language_model.layers
            if min(layers) < 0 or max(layers) >= len(decoder):
                raise ValueError(f"requested decoder blocks out of range: {layers}")
            blocks = [decoder[i] for i in layers]
            for block in blocks:
                hooks.callback(block.register_forward_hook(record).remove)
        output = target(**inputs, logits_to_keep=keep, use_cache=False)
    hidden = (
        torch.cat([captured[block][0] for block in blocks], dim=-1).cpu()
        if blocks
        else None
    )
    return output.logits[0], hidden


@torch.inference_mode()
def capture_text_features(target, inputs: Batch, target_layer_ids: Sequence[int]):
    """A text-only row needs one teacher pass and has no image KL."""
    _, hidden = _capture(target, inputs, 1, target_layer_ids)
    return hidden


@torch.inference_mode()
def visual_kl_per_token(
    target,
    seen_inputs: Batch,
    blind_inputs: Batch,
    loss_mask: Sequence[int],
    *,
    target_layer_ids: Sequence[int] = (),
):
    """Return compact CPU fp32 KL, with-image entropy, and optional [seq, width] features."""
    if (
        seen_inputs.get("pixel_values") is None
        or seen_inputs.get("image_grid_thw") is None
    ):
        raise ValueError("with-image batch requires pixel_values and image_grid_thw")
    if blind_inputs.get("pixel_values") is not None:
        raise ValueError("blind batch contains pixel_values")
    seen_ids = seen_inputs["input_ids"]
    blind_ids = blind_inputs["input_ids"]
    if (
        seen_ids.ndim != 2
        or blind_ids.ndim != 2
        or seen_ids.size(0) != 1
        or blind_ids.size(0) != 1
    ):
        raise ValueError("visual KL expects one example per forward")
    positions = [i for i, value in enumerate(loss_mask) if value]
    offset = seen_ids.size(1) - blind_ids.size(1)
    if len(loss_mask) != seen_ids.size(1) or not positions or offset <= 0:
        raise ValueError("cannot align an empty mask or non-shorter blind sequence")
    first_logit = positions[0] - 1
    if first_logit < offset or not torch.equal(
        seen_ids[:, first_logit:], blind_ids[:, first_logit - offset :]
    ):
        raise ValueError("image and blind response prefixes do not align")

    keep = seen_ids.size(1) - first_logit
    seen, hidden = _capture(target, seen_inputs, keep, target_layer_ids)
    blind, _ = _capture(target, blind_inputs, keep, ())
    indices = torch.tensor(positions, device=seen.device) - 1 - first_logit
    kl, entropy = [], []
    for chunk in indices.split(64):
        log_p = torch.log_softmax(seen[chunk].float(), dim=-1)
        log_q = torch.log_softmax(blind[chunk].float(), dim=-1)
        p = log_p.exp()
        kl.append((p * (log_p - log_q)).sum(-1))
        entropy.append(-(p * log_p).sum(-1))
    kl = torch.cat(kl).cpu()
    entropy = torch.cat(entropy).cpu()
    if not torch.isfinite(kl).all() or not torch.isfinite(entropy).all():
        raise ValueError("target produced non-finite visual KL or entropy")
    return kl, entropy, hidden
