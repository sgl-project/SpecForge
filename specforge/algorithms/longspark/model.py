"""LongSpark objective, training wrapper and train strategy."""

import hashlib

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from specforge.training.strategies.base import DraftTrainStrategy, StepOutput

from . import FEATURES
from .sampling import sample_anchors


def objective(draft_logits, teacher_logits, labels, *, ce_weight, l1_weight, gamma):
    """Position-weighted ``ce_weight * CE + l1_weight * L1(teacher, draft)``.

    CE uses the dataset labels; L1 is the distance between the teacher and draft
    probability distributions over the full vocabulary. Position ``k`` (0-based)
    in the draft block is weighted by ``exp(-k / gamma)``, or uniformly when
    ``gamma`` is ``None``.
    """
    if (
        draft_logits.shape != teacher_logits.shape
        or draft_logits.shape[:-1] != labels.shape
    ):
        raise ValueError("draft, teacher and label shapes disagree")
    if gamma is not None and gamma <= 0:
        raise ValueError("loss_decay_gamma must be positive")
    ce = F.cross_entropy(
        draft_logits.float().reshape(-1, draft_logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    ).reshape_as(labels)
    p = F.softmax(teacher_logits.float(), dim=-1).detach()
    q = F.softmax(draft_logits.float(), dim=-1)
    l1 = (q - p).abs().sum(dim=-1)
    positions = torch.arange(labels.shape[1], device=labels.device, dtype=torch.float32)
    if gamma is None:
        weights = torch.ones_like(positions)[None, :].expand_as(ce)
    else:
        weights = torch.exp(-positions / gamma)[None, :].expand_as(ce)
    numerator = ((ce_weight * ce + l1_weight * l1) * weights).sum()
    denominator = weights.sum()
    with torch.no_grad():
        correct = draft_logits.argmax(dim=-1).eq(labels).float()
        accepted = correct.cumprod(dim=1).sum(dim=1)
        n = correct.new_tensor(labels.shape[0])
        ratios = {
            "hard_ce": ((ce * weights).sum(), denominator),
            "probability_l1": ((l1 * weights).sum(), denominator),
            "accuracy": (correct.sum(), correct.new_tensor(correct.numel())),
            "prefix_acceptance": (accepted.sum(), n),
        }
        for position in range(labels.shape[1]):
            ratios[f"accuracy_position_{position + 1}"] = (
                correct[:, position].sum(),
                n,
            )
            ratios[f"overlap_position_{position + 1}"] = (
                (1.0 - 0.5 * l1[:, position]).sum(),
                n,
            )
    return numerator, denominator.detach(), ratios


class FrozenFinalNorm(nn.Module):
    def __init__(self, weight, epsilon):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.variance_epsilon = float(epsilon)

    def forward(self, value):
        return F.rms_norm(value, (value.shape[-1],), self.weight, self.variance_epsilon)


class LongSparkTrainingModel(nn.Module):
    def __init__(
        self,
        draft_model,
        embed_tokens,
        lm_head,
        final_norm,
        *,
        num_anchors,
        ce_weight,
        l1_weight,
        gamma,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.embed_tokens = embed_tokens.eval().requires_grad_(False)
        self.lm_head = lm_head.eval().requires_grad_(False)
        self.final_norm = final_norm.eval().requires_grad_(False)
        self.num_anchors = num_anchors
        self.ce_weight = ce_weight
        self.l1_weight = l1_weight
        self.gamma = gamma

    def forward(
        self,
        input_ids,
        loss_mask,
        hidden_states,
        target_last_hidden_states,
        target_keys,
        target_values,
        anchor_generator=None,
    ):
        """Return ``(loss numerator, loss denominator, metric ratios)``.

        Capture tensors are token-major: K/V are
        ``[batch, tokens, layers, kv_heads, head_dim]``.
        """
        config = self.draft_model.config
        batch, length = input_ids.shape
        expected_kv = (
            batch,
            length,
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
        )
        if target_keys.shape != expected_kv or target_values.shape != expected_kv:
            raise ValueError(f"target K/V must be token-major with shape {expected_kv}")
        if hidden_states.shape != (
            batch,
            length,
            config.num_hidden_layers * config.hidden_size,
        ):
            raise ValueError(
                "selected target hidden-state shape does not match the draft configuration"
            )
        if target_last_hidden_states.shape != (batch, length, config.hidden_size):
            raise ValueError(
                "teacher final hidden states must be post-norm [batch, tokens, hidden]"
            )
        anchors = sample_anchors(
            input_ids,
            loss_mask,
            num_anchors=self.num_anchors,
            block_size=config.block_size,
            generator=anchor_generator,
        )
        source, position = anchors.source_indices, anchors.positions
        selected_hidden = hidden_states.reshape(
            batch, length, config.num_hidden_layers, config.hidden_size
        )[source, position - 1].detach()
        states = self.draft_model(
            keys=target_keys.detach().permute(0, 2, 3, 1, 4),
            values=target_values.detach().permute(0, 2, 3, 1, 4),
            source_indices=source,
            anchor_positions=position,
            current_hidden_states=selected_hidden,
            current_token_ids=anchors.current_tokens,
            target_embed_tokens=self.embed_tokens,
        )
        previous = torch.cat(
            [anchors.current_tokens[:, None], anchors.labels[:, :-1]], dim=1
        )
        logits = self.draft_model.project_logits(
            states,
            target_final_norm=self.final_norm,
            target_lm_head=self.lm_head,
            previous_token_ids=previous,
        )
        with torch.no_grad():
            teacher_positions = (
                position[:, None]
                + torch.arange(config.block_size, device=input_ids.device)[None, :]
            )
            teacher_hidden = target_last_hidden_states[
                source[:, None], teacher_positions
            ]
            teacher_logits = F.linear(
                teacher_hidden.to(self.lm_head.weight.dtype), self.lm_head.weight
            ).float()
        return objective(
            logits,
            teacher_logits,
            anchors.labels,
            ce_weight=self.ce_weight,
            l1_weight=self.l1_weight,
            gamma=self.gamma,
        )


class LongSparkTrainStrategy(DraftTrainStrategy):
    name = "longspark"
    required_features = set(FEATURES)

    def __init__(self, model, *, seed=0):
        self.model = model
        self.seed = seed

    def trainable_module(self):
        return self.model

    def forward_loss(self, batch, ctx=None):
        self.validate_batch(batch)
        device = next(self.model.parameters()).device
        tensors = {
            key: batch.tensors[key].to(device, non_blocking=True) for key in FEATURES
        }
        if self.model.training and (ctx is None or ctx.epoch_microstep is None):
            raise ValueError(
                "LongSpark requires epoch/microstep context for reproducible anchors"
            )
        rank = dist.get_rank() if dist.is_initialized() else 0
        epoch = ctx.epoch if ctx is not None else 0
        microstep = ctx.epoch_microstep if ctx is not None else None
        seed_parts = f"{self.seed}:{epoch}:{rank}:{microstep or 0}".encode("ascii")
        seed = int.from_bytes(hashlib.sha256(seed_parts).digest()[:8], "big") & (
            (1 << 63) - 1
        )
        generator = torch.Generator(device=device).manual_seed(seed)
        numerator, denominator, ratios = self.model(
            **tensors, anchor_generator=generator
        )
        # Normalize across DP ranks for each micro-batch, then average over
        # gradient accumulation (rather than normalizing once per optimizer
        # step, which would reweight shorter rows).
        global_denominator = denominator.detach().clone()
        world_size = 1
        # Evaluation shards can have unequal lengths. The evaluator reduces
        # additive metrics after each rank finishes; never introduce an
        # extra per-batch collective into that path.
        if self.model.training and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                dist.all_reduce(global_denominator)
        loss = world_size * numerator / global_denominator.clamp_min(1.0)
        # Report the token-weighted loss alongside the per-micro-batch loss
        # that is actually optimized.
        ratios["loss"] = (numerator.detach(), denominator)
        ratios["optimizer_loss"] = (loss.detach(), denominator.new_ones(()))
        return StepOutput(
            loss=loss,
            metrics={},
            ratio_metrics=ratios,
        )

    def checkpoint_state_filter(self, state_dict):
        return {
            key.removeprefix("draft_model."): value
            for key, value in state_dict.items()
            if key.startswith("draft_model.")
        }
