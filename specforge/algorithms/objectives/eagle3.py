"""EAGLE recurrence, teacher preparation and loss policy over model-owned execution."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from specforge.algorithms.losses.eagle import eagle_loss
from specforge.algorithms.losses.eagle_teacher import (
    build_trim_pack,
    compute_target_p_padded,
)
from specforge.core.compact_teacher import compute_target_p_padded_from_hidden
from specforge.modeling.draft.bindings import BoundLMHead

from .base import ObjectiveResult, check_options
from .inputs import valid_tokens
from .registry import register_objective


def shift_left(tensor):
    return torch.cat((tensor[:, 1:], torch.zeros_like(tensor[:, :1])), dim=1)


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
    lk_loss_type: str | None = None
    kl_scale: float = 1.0
    kl_decay: float = 1.0
    compact_teacher: bool = False
    compact_teacher_chunk_size: int = 32768
    trim_loss_positions: bool = False

    def __post_init__(self):
        check_options(
            counts={
                "ttt_length": self.ttt_length,
                "compact_teacher_chunk_size": self.compact_teacher_chunk_size,
            },
            probabilities={"depth_decay": self.depth_decay},
            nonnegative={"kl_scale": self.kl_scale, "kl_decay": self.kl_decay},
        )
        if self.lk_loss_type not in (None, "alpha", "lambda", "tv"):
            raise ValueError("lk_loss_type must be None, alpha, lambda, or tv")
        for name in ("compact_teacher", "trim_loss_positions"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")

    def __call__(self, runtime, batch, **context):
        model, ids = runtime.draft_model, batch["input_ids"]
        model.vocabulary.require_ready()
        valid = valid_tokens(batch)
        backend = getattr(
            runtime, "attention_backend", model.backbone.layers[0].self_attn.backend
        )
        # USP input preparation supplies an overlap tail for deeper teachers.
        length = ids.shape[1] - (self.ttt_length if backend == "usp" else 0)
        if length < 1:
            raise ValueError("EAGLE needs at least one owned sequence position")
        mask = shift_left(valid).float()[..., None]
        tokens = shift_left(ids)
        hidden = model.fusion(batch["hidden_states"])[:, :length]
        positions = batch.get("position_ids")
        if positions is None:
            positions = torch.arange(length, device=ids.device)[None].expand(
                ids.shape[0], -1
            )
        positions = positions[..., :length]
        padding = batch.get("attention_mask", torch.ones_like(ids))
        if backend != "usp":
            padding = padding[:, :length]
        rollout = model.new_rollout(
            backend,
            ring_group=getattr(runtime, "ring_group", None),
            ulysses_group=getattr(runtime, "ulysses_group", None),
        )
        teacher_hidden = shift_left(batch["target_last_hidden_states"])
        trim = None
        with torch.no_grad():
            if self.compact_teacher:
                head = runtime.target_head
                if head is None:
                    raise ValueError("compact_teacher requires a frozen target head")
                tables = compute_target_p_padded_from_hidden(
                    teacher_hidden,
                    (
                        head.bound_weight()
                        if isinstance(head, BoundLMHead)
                        else head.weight
                    ),
                    model.vocabulary.t2d,
                    mask,
                    self.ttt_length,
                    chunk_size=self.compact_teacher_chunk_size,
                )
            else:
                teacher = (
                    shift_left(batch["target_logits"])
                    if "target_logits" in batch
                    else runtime.target_head(teacher_hidden)
                )
                if (
                    self.trim_loss_positions
                    and self.lk_loss_type is None
                    and ids.shape[0] == 1
                ):
                    trim = build_trim_pack(
                        teacher,
                        model.vocabulary.t2d,
                        mask,
                        self.ttt_length,
                        chunk_len=length,
                    )
                tables = (
                    None
                    if trim
                    else compute_target_p_padded(
                        teacher, model.vocabulary.t2d, mask, self.ttt_length
                    )
                )
        total, ratios = hidden.sum() * 0, {}
        denominator = hidden.new_tensor(ids.shape[0] * length)
        for depth in range(self.ttt_length):
            hidden = rollout.step(tokens[:, :length], hidden, padding, positions)
            if trim is None:
                target, on_draft, teacher_ids, position_mask = tables
                target = target[:, depth : depth + length].contiguous()
                on_draft = on_draft[:, depth : depth + length].contiguous()
                teacher_ids = teacher_ids[:, depth : depth + length]
                pm = F.pad(position_mask[:, depth:], (0, 0, 0, depth))[:, :length]
                lm = mask[:, :length]
                logits = model.compute_logits(hidden)
                scale = 1.0
            else:
                rows, keep = trim["rows_steps"][depth], trim["keep_steps"][depth]
                target, on_draft, teacher_ids, pm, lm = (
                    trim[name].index_select(1, keep)
                    for name in (
                        "target_p_c",
                        "on_draft_c",
                        "token_ids_c",
                        "position_mask_sup",
                        "loss_mask_sup",
                    )
                )
                count = trim["nrows_steps"][depth]
                if count == 0:
                    pm, lm = torch.zeros_like(pm), torch.zeros_like(lm)
                logits = model.compute_logits(hidden.index_select(1, rows))
                scale = count / length
            acceptance, loss = eagle_loss(
                logits=logits,
                target_p=target,
                target_p_on_draft=on_draft,
                position_mask=pm,
                lk_loss_type=self.lk_loss_type,
                kl_scale=self.kl_scale,
                kl_decay=self.kl_decay,
                reduce_metrics_fn=getattr(runtime, "reduce_sequence_metrics", None),
            )
            loss = loss * scale
            total = total + self.depth_decay**depth * loss * denominator
            with torch.no_grad():
                predicted = logits.argmax(-1)
                predicted = predicted + model.vocabulary.d2t[predicted]
                correct = ((predicted == teacher_ids) * lm.squeeze(-1)).sum()
                ratios[f"position_{depth}_accuracy"] = (correct, lm.sum())
                ratios[f"position_{depth}_acceptance"] = (
                    acceptance * pm.sum(),
                    pm.sum(),
                )
                ratios[f"position_{depth}_loss"] = (
                    loss.detach() * denominator,
                    denominator,
                )
            tokens, mask = shift_left(tokens), shift_left(mask)
        return ObjectiveResult(total, denominator, ratios)
