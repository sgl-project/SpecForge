"""EAGLE recurrent distillation with an explicit, differentiable rollout tree."""

from dataclasses import dataclass

import torch

from .base import ObjectiveResult, check_options
from .inputs import gather, sequence_inputs, valid_tokens
from .registry import register_objective


def teacher_distribution(model, logits):
    vocabulary = model.vocabulary
    vocabulary.require_ready()
    keep = vocabulary.t2d[logits.argmax(-1)]
    return logits[..., vocabulary.t2d].float().softmax(-1), keep


@register_objective(
    "eagle3",
    architectures=("Eagle3DraftModel",),
    inputs=("input_ids", "hidden_states", "target_last_hidden_states", "loss_mask"),
    outputs=("logits", "hidden_states"),
    trains=("backbone", "fusion", "embed_tokens", "lm_head"),
)
@dataclass
class Eagle3Objective:
    ttt_length: int = 7
    depth_decay: float = 0.8

    def __post_init__(self):
        check_options(
            counts={"ttt_length": self.ttt_length},
            probabilities={"depth_decay": self.depth_decay},
        )

    def __call__(self, runtime, batch, **context):
        model, ids = runtime.draft_model, batch["input_ids"]
        valid = valid_tokens(batch)
        length = ids.shape[1]
        anchor = torch.arange(length, device=ids.device)
        initial = model.fusion(batch["hidden_states"])
        hidden, prefix, positions = initial, None, []
        total, ratios = initial.sum() * 0, {}
        for depth in range(self.ttt_length):
            original = (anchor + depth + 1).expand(ids.shape[0], -1)
            tokens = gather(ids, original).masked_fill(original >= length, 0)
            positions.append((anchor + depth).expand(ids.shape[0], -1))
            all_anchor = anchor.repeat(depth + 1)
            all_depth = torch.arange(depth + 1, device=ids.device).repeat_interleave(
                length
            )
            visibility = (
                (all_depth[None] == 0) & (anchor[:, None] >= all_anchor[None])
            ) | (anchor[:, None] == all_anchor[None])
            padding = (
                batch.get("attention_mask", torch.ones_like(ids, dtype=torch.bool))
                .bool()
                .repeat(1, depth + 1)
            )
            visibility = (
                visibility[None] & padding[:, None, :] & padding[:, -length:, None]
            )
            output = model(
                tokens,
                None,
                sequence_inputs(
                    model.config.draft,
                    positions[-1],
                    visibility,
                    key_positions=torch.cat(positions, 1),
                ),
                recurrent_hidden=hidden,
                kv_prefix=prefix,
            )
            hidden, logits = output.hidden_states, output.logits
            prefix = output.kv.prepend(prefix)
            teacher, included = teacher_distribution(
                model, runtime.teacher_logits(batch, original)
            )
            weight = gather(valid, original) & (original < length) & included
            ce = -(teacher * logits.float().log_softmax(-1)).sum(-1)
            total = total + self.depth_decay**depth * (ce * weight).sum()
            ratios[f"position_{depth}_accuracy"] = (
                ((logits.argmax(-1) == teacher.argmax(-1)) * weight).sum().detach(),
                weight.sum().detach(),
            )
        # Legacy EAGLE normalizes masked soft CE by all sequence positions.
        denominator = total.new_tensor(ids.numel())
        return ObjectiveResult(total, denominator, ratios)
