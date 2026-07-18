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
geometric sum. The evaluator's own collective schedule is decided globally, so
empty or scalar-only shards issue the same reductions as their peers; when
``forward_fn`` is itself collective (FSDP), every rank must additionally
iterate the same number of eval batches."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
import torch.distributed as dist

from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import StepOutput


class Evaluator:
    """Aggregate a full eval pass into ``eval/*`` metrics.

    ``eval/avg_acc``: for per-position (TTT) strategies, position-0 accuracy
    from eval-set-wide correct/denom counts; for scalar strategies
    (DFlash/Domino), the ``accuracy_denom``-weighted mean of batch accuracy.
    """

    def run(
        self,
        forward_fn: Callable[[TrainBatch], StepOutput],
        batches: Optional[Iterable[TrainBatch]],
    ) -> Dict[str, Any]:
        """Run the pass; returns ``{}`` if zero batches were processed globally.

        Scalar accuracy is weighted by ``metrics['accuracy_denom']`` when present,
        else by the loss-token count — only approximately batch-size invariant
        when the accuracy counts a different token set than the loss. In a mixed
        pass, scalar batches feed avg_loss only; their accuracy is not merged.
        """
        # pp rows: [correct, denom, acceptance_rate*w, ploss*w] per TTT
        # position, float64 so counts stay exact past 2**24.
        pp = None
        # [loss*w, w, scalar_acc*denom, scalar_denom, n_batches, ar_w, pl_w]
        sums = None

        with torch.no_grad():
            for batch in batches if batches is not None else ():
                out = forward_fn(batch)
                m = out.metrics
                # .mean() normalizes a shape-[1] loss to the 0-dim slot.
                loss = (
                    out.loss.detach().double().mean()
                    if isinstance(out.loss, torch.Tensor)
                    else torch.tensor(float(out.loss), dtype=torch.float64)
                )
                if sums is None:
                    sums = torch.zeros(7, dtype=torch.float64, device=loss.device)
                tokens = self._token_count(batch, m, device=sums.device)
                sums[0] += loss.to(sums.device) * tokens
                sums[1] += tokens
                sums[4] += 1.0

                if "acc_corrects" in m and "acc_denoms" in m:
                    correct = self._stack(m["acc_corrects"])
                    denom = self._stack(m["acc_denoms"])
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
                    if "acceptance_rates" in m:
                        pp[2] += self._stack(m["acceptance_rates"]) * w
                        sums[5] += tokens
                    if "plosses" in m:
                        pp[3] += self._stack(m["plosses"]) * w
                        sums[6] += tokens
                elif "accuracy" in m:
                    acc = m["accuracy"]
                    acc = (
                        acc.detach().double().mean().to(sums.device)
                        if isinstance(acc, torch.Tensor)
                        else torch.tensor(
                            float(acc), dtype=torch.float64, device=sums.device
                        )
                    )
                    denom = m.get("accuracy_denom")
                    w = (
                        torch.as_tensor(denom).detach().double().sum().to(sums.device)
                        if denom is not None
                        else tokens
                    )
                    sums[2] += acc * w
                    sums[3] += w

        # Fixed global schedule, identical on every rank: (1) SUM the scalar
        # sums, (2) MAX the per-position length — a global decision, so an
        # empty shard still participates — (3) SUM one padded count buffer iff
        # any rank has per-position data.
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size > 1:
            device = self._comm_device()
            sums = (
                sums if sums is not None else torch.zeros(7, dtype=torch.float64)
            ).to(device)
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            local_len = pp.size(1) if pp is not None else 0
            pp_len = torch.tensor([local_len], dtype=torch.int64, device=device)
            dist.all_reduce(pp_len, op=dist.ReduceOp.MAX)
            global_len = int(pp_len.item())
            if global_len > 0:
                buf = torch.zeros(4, global_len, dtype=torch.float64, device=device)
                if pp is not None:
                    buf[:, : pp.size(1)] = pp.to(device)
                dist.all_reduce(buf, op=dist.ReduceOp.SUM)
                pp = buf

        if sums is None or sums[4].item() == 0.0:
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
    def _comm_device() -> torch.device:
        """Collective device: this rank's BOUND cuda device for NCCL — via
        current_device, not the LOCAL_RANK env var, which non-torchrun
        launchers may not export — else CPU."""
        if "nccl" in str(dist.get_backend()):
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

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
    def _token_count(
        batch: TrainBatch, metrics: Dict[str, Any], device
    ) -> torch.Tensor:
        """This batch's token weight as a 0-dim float64 tensor (no host sync):
        the strategy's loss denoms, else the loss mask, else 1."""
        denoms = metrics.get("metric_loss_denoms")
        if denoms:
            return (
                torch.stack([torch.as_tensor(d).detach().float() for d in denoms])
                .sum()
                .double()
                .to(device)
            )
        loss_mask = batch.tensors.get("loss_mask")
        if isinstance(loss_mask, torch.Tensor):
            return loss_mask.sum().double().to(device)
        return torch.ones((), dtype=torch.float64, device=device)


__all__ = ["Evaluator"]
