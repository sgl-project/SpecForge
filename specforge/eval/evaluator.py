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

Data-parallel eval: **every** reported metric — loss, per-position counts, and
the scalar-accuracy sums — is reduced across ranks, so the numbers cover the
whole eval set regardless of sharding. The evaluator's OWN collective schedule
is decided *globally*, never from rank-local shard content: a rank whose shard
yields only scalar batches — or nothing — issues exactly the same reductions as
its peers. NB this covers the evaluator's reductions, not the caller's
``forward_fn``: when the forward itself is collective (an FSDP-wrapped module
all-gathers parameters per forward), every rank must iterate the SAME NUMBER of
eval batches — shard by equal counts, or hand every rank the same eval set.
Accumulation stays on-device; the single host sync happens after the loop.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List

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
    ) -> Dict[str, Any]:
        per_pos_correct = None  # [ttt_length] float tensor
        per_pos_denom = None
        # [loss*tokens, tokens, scalar_acc_sum, scalar_acc_count], accumulated
        # on the eval device (float64: exact counts, stable loss sums).
        sums = None

        with torch.no_grad():
            for batch in batches:
                out = forward_fn(batch)
                m = out.metrics
                # .mean() normalizes to 0-dim (a shape-[1] loss would not
                # broadcast into the 0-dim accumulator slot).
                loss = (
                    out.loss.detach().double().mean()
                    if isinstance(out.loss, torch.Tensor)
                    else torch.tensor(float(out.loss), dtype=torch.float64)
                )
                if sums is None:
                    sums = torch.zeros(4, dtype=torch.float64, device=loss.device)
                tokens = self._token_count(batch, m, device=sums.device)
                sums[0] += loss.to(sums.device) * tokens
                sums[1] += tokens

                if "acc_corrects" in m and "acc_denoms" in m:
                    correct = torch.stack(
                        [torch.as_tensor(c).detach().float() for c in m["acc_corrects"]]
                    )
                    denom = torch.stack(
                        [torch.as_tensor(d).detach().float() for d in m["acc_denoms"]]
                    )
                    if per_pos_correct is None:
                        per_pos_correct = torch.zeros_like(correct)
                        per_pos_denom = torch.zeros_like(denom)
                    per_pos_correct += correct
                    per_pos_denom += denom
                elif "accuracy" in m:
                    acc = m["accuracy"]
                    acc = (
                        acc.detach().double().mean().to(sums.device)
                        if isinstance(acc, torch.Tensor)
                        else torch.tensor(
                            float(acc), dtype=torch.float64, device=sums.device
                        )
                    )
                    sums[2] += acc
                    sums[3] += 1.0

        # Aggregate across data-parallel ranks (each iterates its own eval shard)
        # so every metric is over the WHOLE eval set. The schedule is fixed and
        # identical on all ranks: (1) SUM the scalar sums, (2) MAX the
        # per-position length — a GLOBAL decision, so an empty or scalar-only
        # shard still participates — then (3) SUM one stacked count buffer iff
        # any rank has per-position data. world_size==1 skips all of it.
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size > 1:
            device = self._comm_device()
            sums = (
                sums if sums is not None else torch.zeros(4, dtype=torch.float64)
            ).to(device)
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            local_len = per_pos_correct.numel() if per_pos_correct is not None else 0
            pp_len = torch.tensor([local_len], dtype=torch.int64, device=device)
            dist.all_reduce(pp_len, op=dist.ReduceOp.MAX)
            global_len = int(pp_len.item())
            if global_len > 0:
                buf = torch.zeros(2, global_len, dtype=torch.float64, device=device)
                if per_pos_correct is not None:
                    n = per_pos_correct.numel()
                    buf[0, :n] = per_pos_correct.double().to(device)
                    buf[1, :n] = per_pos_denom.double().to(device)
                dist.all_reduce(buf, op=dist.ReduceOp.SUM)
                per_pos_correct, per_pos_denom = buf[0], buf[1]
        elif sums is None:
            sums = torch.zeros(4, dtype=torch.float64)

        loss_x_tokens, total_tokens, scalar_acc_sum, scalar_acc_n = sums.tolist()
        avg_loss = loss_x_tokens / max(total_tokens, 1.0)

        if per_pos_correct is not None:
            per_position_acc = (
                per_pos_correct.double().cpu()
                / per_pos_denom.double().cpu().clamp_min(1.0)
            ).tolist()
            return {
                "eval/avg_loss": avg_loss,
                "eval/avg_acc": float(per_position_acc[0]) if per_position_acc else 0.0,
                "eval/per_position_acc": per_position_acc,
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
    def _comm_device() -> torch.device:
        """The device collectives must use: this rank's BOUND CUDA device for NCCL,
        CPU otherwise. ``current_device`` (set by ``init_distributed`` on every
        launch path) rather than the LOCAL_RANK env var, which non-torchrun
        launchers may not export — an unset LOCAL_RANK would put every rank's
        reduction tensor on cuda:0 and break the NCCL communicator."""
        if "nccl" in str(dist.get_backend()):
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

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
    def _token_count(batch: TrainBatch, metrics: Dict[str, Any], device) -> torch.Tensor:
        """This batch's token weight as a 0-dim float64 tensor (no host sync)."""
        # Prefer the loss denom the strategy already computed; fall back to the
        # loss mask, then to a per-batch weight of 1.
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
