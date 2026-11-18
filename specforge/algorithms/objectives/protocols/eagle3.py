"""EAGLE recurrent distillation with an explicit, differentiable rollout tree."""

from dataclasses import dataclass
from functools import partial

import torch

from specforge.algorithms.metrics.rollout import expected_acceptance

from ..base import check_options
from ..diagnostics import measure_predictions
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
        # evaluation reports the expected acceptance of every depth
        evaluation_diagnostics=True,
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
        deferred = [] if context.get("collect_detailed_metrics", True) else None
        return PredictionBatch(
            self.rollout(runtime, batch, deferred),
            reduction="sum_means",
            set_metrics=partial(depth_losses, runtime),
            diagnostics=(
                None
                if deferred is None
                else partial(depth_acceptance, runtime, deferred)
            ),
        )

    def rollout(self, runtime, batch, deferred=None):
        """Yield each depth's predictions as the rollout reaches it.

        A list passed as ``deferred`` receives one builder per depth: the
        diagnostics rebuild a depth from it, so no depth's gathered teacher
        outlives its loss.
        """
        model, ids = runtime.draft_model, batch["input_ids"]
        valid = valid_tokens(batch)
        if "lengths" in batch and batch["lengths"].numel() > ids.shape[0]:
            raise ValueError("EAGLE requires unpacked sequences")
        backend, groups = runtime.attention_backend, runtime.sequence_parallel
        parallel = backend == "usp"
        length = ids.shape[1] - (self.ttt_length if parallel else 0)
        if length < 1:
            raise ValueError("EAGLE requires at least one owned sequence position")
        cached = (
            model.new_rollout(
                backend,
                ring_group=groups.ring_group if groups else None,
                ulysses_group=groups.ulysses_group if groups else None,
            )
            if backend in ("fa", "usp", "flex_attention")
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
            build = partial(
                runtime.predictions,
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
                # every depth also counts toward the pooled ``acc``
                accuracy_name=f"acc_{depth}",
                reduce_statistics=groups.reduce_statistics if groups else None,
            )
            if deferred is not None:
                deferred.append(build)
            yield build()


def depth_losses(runtime, results):
    """Name what each depth's loss computed as the training wrappers did.

    ``ploss_<depth>`` is the depth's mean loss before its depth weight, which
    moves to the denominator here; a depth the weighting zeroes reports zero.
    The same ratios, and their weighted sum, carry the token loss's own name:
    ``kl_loss`` for soft cross entropy and ``lk_loss`` for an LK loss.
    """
    kind = runtime.loss_configuration.terms["token"].loss.kind
    name = (
        "lk_loss"
        if kind.startswith("lk_")
        else "kl_loss" if kind == "soft_ce" else f"{kind}_loss"
    )
    ratios, total = {}, 0
    for depth, result in enumerate(results):
        term = result.terms[0]
        numerator = term.numerator.detach()
        weight = float(runtime.weighting(torch.ones(()), torch.tensor(depth)))
        pair = (numerator, term.denominator * weight)
        ratios[f"ploss_{depth}"] = ratios[f"{name}_{depth}"] = pair
        total = total + numerator
    ratios[name] = (total, results[0].terms[0].denominator)
    return ratios


def depth_acceptance(runtime, deferred):
    """Measure the expected acceptance of every rollout depth.

    The wrappers reported each depth's mean over its supervised positions and
    weighted it by the loss's token count wherever depths, micro-batches or
    ranks are combined. The pairs returned here do the same.
    """
    ratios, accepted, tokens = {}, 0, 0
    for depth, build in enumerate(deferred):
        predictions = build()
        measured = measure_predictions(runtime, predictions, expected_acceptance)
        rate = measured["accepted"] / measured["supervised"].clamp_min(1e-8)
        pair = (rate * predictions.denominator, predictions.denominator)
        ratios[f"acceptance_rate_{depth}"] = ratios[f"expected_acceptance_{depth}"] = (
            pair
        )
        accepted, tokens = accepted + pair[0], tokens + pair[1]
    ratios["acceptance_rate"] = ratios["expected_acceptance"] = (accepted, tokens)
    return ratios, {}
