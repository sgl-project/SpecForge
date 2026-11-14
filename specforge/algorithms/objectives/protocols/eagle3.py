"""EAGLE recurrent distillation with an explicit, differentiable rollout tree."""

from dataclasses import dataclass

import torch

from ..base import check_options
from ..inputs import gather, sequence_inputs, valid_tokens
from ..losses import TokenLoss
from ..predictions import PredictionBatch
from ..registry import TrainingRequirements, register_protocol


@register_protocol(
    "eagle3",
    training=TrainingRequirements(
        capture_method="eagle3",
        supports_vocab_mapping=True,
        attention_backends=frozenset({"eager", "sdpa", "flex_attention", "fa", "usp"}),
        supports_sequence_parallel=True,
    ),
    requires=("teacher_logits",),
    architectures=("Eagle3DraftModel",),
    inputs=("input_ids", "hidden_states", "target_last_hidden_states", "loss_mask"),
    provides=(
        "logits",
        "labels",
        "teacher_logits",
        "chain_positions",
        "acceptance_chain",
    ),
    trains=("backbone", "fusion", "embed_tokens", "lm_head"),
)
@dataclass
class Eagle3Objective:
    ttt_length: int = 7
    depth_decay: float = 0.8

    default_loss = TokenLoss(kind="soft_ce")
    default_weighting_options = ("depth_decay",)

    def __post_init__(self):
        check_options(
            counts={"ttt_length": self.ttt_length},
            probabilities={"depth_decay": self.depth_decay},
        )

    @property
    def default_weighting(self):
        return {"kind": "geometric", "factor": self.depth_decay}

    def __call__(self, runtime, batch, **context):
        return PredictionBatch(self.rollout(runtime, batch), reduction="sum_means")

    def rollout(self, runtime, batch):
        model, ids = runtime.draft_model, batch["input_ids"]
        valid = valid_tokens(batch)
        if "lengths" in batch and batch["lengths"].numel() > ids.shape[0]:
            raise ValueError("EAGLE requires unpacked sequences")
        backend = getattr(runtime, "attention_backend", None)
        parallel = backend == "usp"
        length = ids.shape[1] - (self.ttt_length if parallel else 0)
        if length < 1:
            raise ValueError("EAGLE requires at least one owned sequence position")
        cached = (
            model.new_rollout(
                backend,
                ring_group=getattr(runtime, "ring_group", None),
                ulysses_group=getattr(runtime, "ulysses_group", None),
            )
            if backend in ("fa", "usp")
            else None
        )
        anchor = torch.arange(length, device=ids.device)
        initial = model.fusion(batch["hidden_states"])[:, :length]
        hidden, prefix, positions = initial, None, []
        for depth in range(self.ttt_length):
            original = (anchor + depth + 1).expand(ids.shape[0], -1)
            tokens = gather(ids, original).masked_fill(original >= ids.shape[1], 0)
            positions.append((anchor + depth).expand(ids.shape[0], -1))
            if cached is not None:
                padding = batch.get("attention_mask", torch.ones_like(ids))
                rope_positions = batch.get("position_ids", anchor[None])[..., :length]
                hidden = cached.step(tokens, hidden, padding, rope_positions)
            else:
                all_anchor = anchor.repeat(depth + 1)
                all_depth = torch.arange(
                    depth + 1, device=ids.device
                ).repeat_interleave(length)
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
                        rotary_positions=(
                            batch["position_ids"] + depth
                            if "position_ids" in batch
                            else None
                        ),
                    ),
                    recurrent_hidden=hidden,
                    kv_prefix=prefix,
                    output_logits=False,
                )
                hidden = output.hidden_states
                prefix = output.kv.prepend(prefix)
            model.vocabulary.require_ready()
            supervised = gather(valid, original) & (original < ids.shape[1])
            yield runtime.predictions(
                hidden,
                batch,
                original,
                mask=supervised,
                positions=torch.tensor(depth, device=ids.device),
                vocab_mask=model.vocabulary.t2d,
                draft_to_target=model.vocabulary.d2t,
                requires=frozenset({"teacher_logits"}),
                default_loss=self.default_loss,
                denominator=torch.tensor(
                    float(ids.shape[0] * length), device=ids.device
                ),
                accuracy_name=f"position_{depth}_accuracy",
                report_accuracy=depth == 0,
                reduce_statistics=getattr(runtime, "reduce_sequence_statistics", None),
            )
