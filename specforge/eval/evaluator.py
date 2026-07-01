# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Evaluator: acceptance-length and per-position accuracy over an eval pass.

The one thing scalar-averaging gets wrong for speculative decoding: per-position
accuracy must be aggregated *across the whole eval set first* — summing the
per-position correct/denom counts — and only *then* folded into the geometric
sum. Averaging each batch's ``simulated_acc_len`` (or treating a batch's
per-position vector as if it were the eval set's) makes the number depend on the
eval batch size. Summing counts first is batch-size invariant.

EAGLE3 emits per-position ``acc_corrects`` / ``acc_denoms`` (one scalar per TTT
position); DFlash / Domino emit a single scalar ``accuracy`` and have no
per-position structure, so their ``simulated_acc_len`` degenerates to that scalar.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
import torch.distributed as dist

from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import StepOutput


class Evaluator:
    """Aggregate a full eval pass into ``{avg_loss, avg_acc, simulated_acc_len}``."""

    def run(
        self,
        forward_fn: Callable[[TrainBatch], StepOutput],
        batches: Iterable[TrainBatch],
    ) -> Dict[str, float]:
        per_pos_correct: Optional[torch.Tensor] = None  # [ttt_length]
        per_pos_denom: Optional[torch.Tensor] = None
        loss_x_tokens = 0.0
        total_tokens = 0
        scalar_acc_sum, scalar_acc_n = 0.0, 0

        with torch.no_grad():
            for batch in batches:
                out = forward_fn(batch)
                m = out.metrics
                tokens = self._token_count(batch, m)
                loss_x_tokens += float(out.loss) * tokens
                total_tokens += tokens

                if "acc_corrects" in m and "acc_denoms" in m:
                    correct = torch.stack(
                        [c.detach().float() for c in m["acc_corrects"]]
                    )
                    denom = torch.stack([d.detach().float() for d in m["acc_denoms"]])
                    if per_pos_correct is None:
                        per_pos_correct = torch.zeros_like(correct)
                        per_pos_denom = torch.zeros_like(denom)
                    per_pos_correct += correct
                    per_pos_denom += denom
                elif "accuracy" in m:
                    scalar_acc_sum += float(m["accuracy"])
                    scalar_acc_n += 1

        # Aggregate across data-parallel ranks (each iterates its own eval shard)
        # so every metric is over the WHOLE eval set — loss the same way as the
        # accuracy counts, matching the legacy trainer. world_size==1 makes the
        # reduction an identity, so skip it (and avoid an all_reduce on the CPU
        # tensors a single-process eval uses).
        if dist.is_initialized() and dist.get_world_size() > 1:
            device = (
                per_pos_correct.device
                if per_pos_correct is not None
                else torch.device("cuda", torch.cuda.current_device())
            )
            totals = torch.tensor(
                [loss_x_tokens, float(total_tokens)], dtype=torch.float64, device=device
            )
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            loss_x_tokens, total_tokens = float(totals[0]), float(totals[1])
            if per_pos_correct is not None:
                dist.all_reduce(per_pos_correct, op=dist.ReduceOp.SUM)
                dist.all_reduce(per_pos_denom, op=dist.ReduceOp.SUM)

        avg_loss = loss_x_tokens / max(total_tokens, 1)

        if per_pos_correct is not None:
            per_position_acc = (per_pos_correct / per_pos_denom.clamp_min(1.0)).tolist()
            return {
                "eval/avg_loss": avg_loss,
                "eval/avg_acc": float(per_position_acc[0]) if per_position_acc else 0.0,
                "eval/simulated_acc_len": self._simulated_acc_len(per_position_acc),
            }

        # Scalar strategies: no per-position vector to geometric-sum.
        avg_acc = scalar_acc_sum / scalar_acc_n if scalar_acc_n else 0.0
        return {
            "eval/avg_loss": avg_loss,
            "eval/avg_acc": avg_acc,
            "eval/simulated_acc_len": avg_acc,
        }

    @staticmethod
    def _simulated_acc_len(per_position_acc: List[float]) -> float:
        """E[accepted tokens] = a0 + a0*a1 + a0*a1*a2 + ... over the aggregated acc.

        ``per_position_acc`` is the eval-set-wide per-position accuracy (length =
        ttt_length), not a list of per-batch vectors.
        """
        cumulative, total = 1.0, 0.0
        for acc in per_position_acc:
            cumulative *= acc
            total += cumulative
        return total

    @staticmethod
    def _token_count(batch: TrainBatch, metrics: Dict[str, Any]) -> int:
        # Prefer the loss denom the strategy already computed; fall back to the
        # loss mask, then to a per-batch weight of 1.
        denoms = metrics.get("metric_loss_denoms")
        if denoms:
            return int(sum(float(d) for d in denoms))
        loss_mask = batch.tensors.get("loss_mask")
        if isinstance(loss_mask, torch.Tensor):
            return int(loss_mask.sum().item())
        return 1


__all__ = ["Evaluator"]
