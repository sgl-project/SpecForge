# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Batch-size-invariant eval aggregation: per-position correct/denom counts are
summed over the whole eval set (all batches, all DP ranks) BEFORE any ratio or
geometric sum. The pass is pooled once, through the step runner, on a schedule
that is the same on every rank, so empty or scalar-only shards issue the same
reductions as their peers; when the runner's forward is itself collective
(FSDP), every rank must additionally iterate the same number of eval batches."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

import torch

from specforge.runtime.contracts import TrainBatch

if TYPE_CHECKING:
    from specforge.training.contracts import StepContext, StepRunner


def depth_ratios(ratios, prefix):
    """Ratio pairs named ``<prefix>_<depth>``, in depth order."""
    start = len(prefix) + 1
    depths = sorted(
        int(name[start:])
        for name in ratios
        if name.startswith(f"{prefix}_") and name[start:].isdigit()
    )
    return [ratios[f"{prefix}_{depth}"] for depth in depths]


class Evaluator:
    """Aggregate a full eval pass into ``eval/*`` metrics.

    ``eval/avg_acc``: for an objective that reports ``acc_<depth>`` pairs,
    position-0 accuracy from eval-set-wide correct/denom counts; otherwise the
    eval-set-wide ratio of its ``acc`` pair.
    """

    def run(
        self,
        runner: "StepRunner",
        batches: Optional[Iterable[TrainBatch]],
        ctx: "StepContext",
    ) -> Dict[str, Any]:
        """Run the pass; returns ``{}`` if zero batches were processed globally.

        Each batch is evaluated by ``runner.evaluate`` inside one
        ``runner.evaluation()`` pass, which is then pooled across ranks by
        ``runner.reduce_sum`` and ``runner.reduce_max``.

        The loss is the ratio of the pass's summed numerators and denominators.
        In a pass that mixes per-depth and single accuracy pairs, the batches
        with a single pair feed avg_loss only.
        """
        # pp rows: [correct, denom, acceptance_rate*w, ploss*w] per TTT
        # position, float64 so counts stay exact past 2**24.
        pp = None
        # [loss*w, w, scalar_acc*denom, scalar_denom, n_batches, ar_w, pl_w]
        sums = None

        # Entered on every rank, with or without batches of its own.
        with runner.evaluation(), torch.no_grad():
            for batch in batches if batches is not None else ():
                result = runner.evaluate(batch, ctx)
                if sums is None:
                    sums = torch.zeros(
                        7, dtype=torch.float64, device=result.numerator.device
                    )
                sums[0] += self._sum64(result.numerator, sums.device)
                sums[1] += self._sum64(result.denominator, sums.device)
                sums[4] += 1.0

                counts = depth_ratios(result.ratios, "acc")
                if counts:
                    # Each depth's mean, weighted by the batch's loss tokens,
                    # which the acceptance pairs carry.
                    acceptance = depth_ratios(result.ratios, "acceptance_rate")
                    losses = depth_ratios(result.ratios, "ploss")
                    tokens = self._token_count(batch, acceptance, device=sums.device)
                    correct = self._stack(n for n, _ in counts)
                    denom = self._stack(d for _, d in counts)
                    if pp is None:
                        pp = torch.zeros(
                            4,
                            correct.numel(),
                            dtype=torch.float64,
                            device=correct.device,
                        )
                    pp[0] += correct
                    pp[1] += denom
                    w = tokens.to(pp.device)
                    if acceptance:
                        pp[2] += self._stack(n / d for n, d in acceptance) * w
                        sums[5] += tokens
                    if losses:
                        pp[3] += (
                            self._stack(n / d.where(d > 0, 1) for n, d in losses) * w
                        )
                        sums[6] += tokens
                elif "acc" in result.ratios:
                    accuracy_numerator, accuracy_denominator = result.ratios["acc"]
                    sums[2] += self._sum64(accuracy_numerator, sums.device)
                    sums[3] += self._sum64(accuracy_denominator, sums.device)

        # Fixed global schedule, identical on every rank: (1) SUM the scalar
        # sums, (2) MAX the per-position length — a global decision, so an
        # empty shard still participates — (3) SUM one padded count buffer iff
        # any rank has per-position data.
        sums = runner.reduce_sum(
            sums if sums is not None else torch.zeros(7, dtype=torch.float64)
        )
        local_len = pp.size(1) if pp is not None else 0
        global_len = int(
            runner.reduce_max(torch.tensor([local_len], dtype=torch.int64)).item()
        )
        if global_len > 0:
            buf = torch.zeros(4, global_len, dtype=torch.float64, device=sums.device)
            if pp is not None:
                buf[:, : pp.size(1)] = pp.to(sums.device)
            pp = runner.reduce_sum(buf)

        if sums[4].item() == 0.0:
            # Zero batches globally: report nothing — fabricated zero metrics
            # would poison best-checkpoint tracking.
            return {}

        loss_x_w, loss_w, acc_sum, acc_w, _n, ar_w, pl_w = sums.tolist()
        avg_loss = loss_x_w / max(loss_w, 1.0)

        if pp is not None:
            pp = pp.cpu()
            per_position_acc = (pp[0] / pp[1].clamp_min(1.0)).tolist()
            metrics = {
                "eval/avg_loss": avg_loss,
                "eval/avg_acc": float(per_position_acc[0]),
                "eval/per_position_acc": per_position_acc,
                "eval/simulated_acc_len": self._simulated_acc_len(per_position_acc),
            }
            if ar_w > 0:
                for i, v in enumerate((pp[2] / ar_w).tolist()):
                    metrics[f"eval/acceptance_rate_{i}"] = v
                    metrics[f"eval/expected_acceptance_{i}"] = v
            if pl_w > 0:
                for i, v in enumerate((pp[3] / pl_w).tolist()):
                    metrics[f"eval/ploss_{i}"] = v
            return metrics

        avg_acc = acc_sum / acc_w if acc_w else 0.0
        return {
            "eval/avg_loss": avg_loss,
            "eval/avg_acc": avg_acc,
            "eval/simulated_acc_len": avg_acc,
        }

    @staticmethod
    def _stack(values: Iterable[Any]) -> torch.Tensor:
        return torch.stack([torch.as_tensor(v).detach().double() for v in values])

    @staticmethod
    def _sum64(value: Any, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(value).detach().double().sum().to(device)

    @staticmethod
    def _simulated_acc_len(per_position_acc: List[float]) -> float:
        """E[accepted tokens] = a0 + a0*a1 + ... over the eval-set-wide
        per-position accuracy (length = ttt_length)."""
        cumulative, total = 1.0, 0.0
        for acc in per_position_acc:
            cumulative *= acc
            total += cumulative
        return total

    @staticmethod
    def _token_count(batch: TrainBatch, acceptance, device) -> torch.Tensor:
        """This batch's token weight as a 0-dim float64 tensor (no host sync):
        the acceptance pairs' denominators, else the loss mask, else 1."""
        if acceptance:
            return (
                torch.stack(
                    [torch.as_tensor(d).detach().float() for _, d in acceptance]
                )
                .sum()
                .double()
                .to(device)
            )
        loss_mask = batch.tensors.get("loss_mask")
        if isinstance(loss_mask, torch.Tensor):
            return loss_mask.sum().double().to(device)
        return torch.ones((), dtype=torch.float64, device=device)


__all__ = ["Evaluator"]
