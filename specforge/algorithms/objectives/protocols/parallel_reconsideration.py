"""Parallel Reconsideration's two rollouts and gated CE (arXiv:2611.05724)."""

from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.nn.functional import all_reduce
from torch.nn import functional as F

from specforge.config.draft import DraftConfig
from specforge.distributed import get_dp_group
from specforge.modeling.draft.layers.attention.kv import AttentionKV
from specforge.modeling.draft.layers.inputs import LayerInputs
from specforge.modeling.draft.models.eagle3 import Eagle3Output

from ..base import ObjectiveResult, check_options
from ..inputs import gather, sequence_inputs, valid_tokens
from ..losses import TokenLoss
from ..predictions import PredictionBatch
from ..registry import TrainingRequirements, register_protocol


def halt_probabilities(logits: Tensor) -> Tensor:
    """Convert [..., depth_cap] halt logits to [..., depth_cap + 1] masses."""
    halt = logits.float().sigmoid()
    terminal = torch.ones_like(halt[..., :1])
    survive = torch.cat((terminal, (1 - halt).cumprod(-1)), -1)
    return torch.cat((halt, terminal), -1) * survive


def reconsideration_token_loss(ce: Tensor, mass: Tensor) -> Tensor:
    """Apply Eq. (6)'s token terms on aligned [..., depth] tensors."""
    deeper = (mass[..., 1:].detach() * ce[..., 1:]).sum(-1)
    gate = (mass * ce.detach()).sum(-1)
    return ce[..., 0] + deeper + gate


def select_halt(
    outputs: list[Eagle3Output], depths: Tensor
) -> tuple[Tensor, AttentionKV]:
    """Gather the feature and rotated KV at each [batch, origin] halt depth."""
    features = torch.stack([out.hidden_states for out in outputs], dim=2)
    hidden = features.gather(
        2, depths[..., None, None].expand(-1, -1, 1, features.shape[-1])
    ).squeeze(2)

    def select_cache(entries: list[Tensor]) -> Tensor:
        cache = torch.stack(entries, dim=3)
        index = depths[:, None, :, None, None].expand(
            -1, cache.shape[1], -1, 1, cache.shape[-1]
        )
        return cache.gather(3, index).squeeze(3)

    return hidden, AttentionKV(
        select_cache([out.kv.key for out in outputs]),
        select_cache([out.kv.value for out in outputs]),
    )


def rollout_inputs(
    config: DraftConfig,
    padding: Tensor,
    positions: Tensor,
    step: int,
    *,
    mixed: bool,
) -> tuple[LayerInputs, ...]:
    """Build a shared base prefix and independent per-origin cache visibility.

    padding is [batch, origin] and positions provides the RoPE indices for
    those rows. Each extra KV block contains one key per origin at a
    previous speculative position. The current block contains the new key.
    """
    length = padding.shape[1]
    anchor = torch.arange(length, device=padding.device).expand_as(padding)
    initial_index = anchor + 1
    base_valid = padding & (initial_index < length) & gather(padding, initial_index)
    current_index = initial_index + step
    active = base_valid & (current_index < length) & gather(padding, current_index)
    origin = anchor[0]
    causal = (
        origin[:, None] > origin[None, :]
        if mixed
        else origin[:, None] >= origin[None, :]
    )
    base = causal & base_valid[:, None, :]
    first = 0 if mixed else 1
    diagonal = (origin[:, None] == origin[None, :]).expand_as(base)
    blocks = [base] + [diagonal for _ in range(first, step + 1)]
    visibility = torch.cat(blocks, dim=-1) & active[..., None]
    return sequence_inputs(
        config,
        anchor + step,
        visibility,
        key_positions=torch.cat(
            [anchor] + [anchor + i for i in range(first, step + 1)], dim=-1
        ),
        rotary_positions=positions + step,
    )


@torch.no_grad()
def _teacher_tokens(runtime, batch) -> Tensor:
    """Read hard teacher IDs in the original unshifted [batch, sequence] layout."""
    if "target_logits" in batch:
        return batch["target_logits"].argmax(-1)
    if runtime.target_head is None or "target_last_hidden_states" not in batch:
        raise ValueError("Parallel Reconsideration requires the frozen teacher head")
    # Resolve teacher tokens once for both rollouts without retaining a full
    # sequence-by-vocabulary tensor. The head includes any target logit transform.
    return torch.cat(
        [
            runtime.target_head(piece).argmax(-1)
            for piece in batch["target_last_hidden_states"].split(128, dim=1)
        ],
        dim=1,
    )


@torch.no_grad()
def _sample_tokens(model, hidden: Tensor) -> Tensor:
    probabilities = model.compute_logits(hidden).float().softmax(-1)
    sampled = torch.multinomial(probabilities.flatten(0, 1), 1).reshape(
        hidden.shape[:2]
    )
    return sampled + model.vocabulary.d2t[sampled]


@register_protocol(
    "parallel_reconsideration",
    training=TrainingRequirements(capture_method="eagle3", supports_vocab_mapping=True),
    requires=("teacher_logits",),
    architectures=("ParallelReconsiderationDraftModel",),
    inputs=("input_ids", "hidden_states", "target_last_hidden_states", "loss_mask"),
    provides=("labels", "logits", "teacher_logits", "chain_positions"),
    trains=("backbone", "fusion", "embed_tokens", "lm_head", "self_query", "halt_gate"),
)
@dataclass
class ParallelReconsiderationObjective:
    ttt_length: int = 7
    max_reconsiderations: int = 2
    balance_beta: float = 0.01

    default_loss = TokenLoss(kind="value")

    def __post_init__(self):
        check_options(
            counts={
                "ttt_length": self.ttt_length,
                "max_reconsiderations": self.max_reconsiderations,
            },
            nonnegative={"balance_beta": self.balance_beta},
        )

    def __call__(self, runtime, batch, **context):
        return PredictionBatch(
            self.rollout(runtime, batch),
            chunk_size=256,
            finalize=partial(self.finalize, synchronize=runtime.draft_model.training),
        )

    def rollout(self, runtime, batch):
        model, ids = runtime.draft_model, batch["input_ids"]
        supervised = valid_tokens(batch)
        padding = batch.get("attention_mask", torch.ones_like(ids)).bool()
        length = ids.shape[1]
        if "lengths" in batch and batch["lengths"].numel() > ids.shape[0]:
            raise ValueError("Parallel Reconsideration requires unpacked sequences")
        model.vocabulary.require_ready()
        selected = model.vocabulary.t2d
        teacher = _teacher_tokens(runtime, batch)
        teacher_labels = (selected.long().cumsum(0) - 1).clamp_min(0)[teacher]
        supervised &= selected[teacher]
        anchor = torch.arange(length, device=ids.device).expand_as(ids)
        positions = batch.get("position_ids", anchor)
        seed_valid = padding & (anchor + 1 < length) & gather(padding, anchor + 1)

        def entries(step, *, mixed):
            return rollout_inputs(
                model.config.draft, padding, positions, step, mixed=mixed
            )

        seed_ids = gather(ids, anchor + 1).masked_fill(anchor + 1 >= length, 0)
        one = mixed = model(
            seed_ids,
            batch["hidden_states"],
            entries(0, mixed=False),
            output_logits=False,
        )
        one_prefix = mixed_prefix = one.kv
        steps = min(self.ttt_length, length - 1)
        for step in range(steps):
            at = anchor + step + 1
            labels = gather(teacher_labels, at)
            mask = gather(supervised, at) & (at < length) & seed_valid
            if step:
                yield self.predict(runtime, one.hidden_states, labels, mask, at)

            reconsider_inputs = entries(step, mixed=True)
            levels = [mixed]
            for _ in range(self.max_reconsiderations):
                levels.append(
                    model.reconsider(
                        levels[-1].hidden_states,
                        reconsider_inputs,
                        kv_prefix=mixed_prefix,
                        output_logits=False,
                    )
                )
            halts = torch.cat(
                [model.halt_gate(out.hidden_states) for out in levels[:-1]], -1
            )
            mass = halt_probabilities(halts)
            yield self.predict(
                runtime, mixed.hidden_states, labels, mask, at, levels=levels, mass=mass
            )
            if step == steps - 1:
                break

            depths = torch.multinomial(mass.detach().flatten(0, 1), 1).reshape(
                ids.shape
            )
            chosen, committed = select_halt(levels, depths)
            mixed_prefix = committed.prepend(mixed_prefix)
            one = model(
                _sample_tokens(model, one.hidden_states),
                None,
                entries(step + 1, mixed=False),
                recurrent_hidden=one.hidden_states,
                kv_prefix=one_prefix,
                output_logits=False,
            )
            one_prefix = one.kv.prepend(one_prefix)
            mixed = model(
                _sample_tokens(model, chosen),
                None,
                entries(step + 1, mixed=True),
                recurrent_hidden=chosen,
                kv_prefix=mixed_prefix,
                output_logits=False,
            )

    def predict(
        self, runtime, hidden, labels, mask, positions, *, levels=None, mass=None
    ):
        model = runtime.draft_model
        context_inputs = (
            {}
            if levels is None
            else {
                "refined": torch.stack(
                    [out.hidden_states for out in levels[1:]], dim=-2
                ),
                "mass": mass,
            }
        )

        def compute(calc):
            def ce(logits):
                return F.cross_entropy(
                    logits.flatten(0, -2).float(),
                    calc.labels.flatten(),
                    reduction="none",
                ).reshape_as(calc.labels)

            initial_ce = ce(calc.outputs()["logits"])
            routed = calc.predictions.context_inputs
            if "mass" in routed:
                losses = torch.stack(
                    [initial_ce]
                    + [
                        ce(model.compute_logits(h))
                        for h in routed["refined"].unbind(-2)
                    ],
                    dim=-1,
                )
                loss = reconsideration_token_loss(losses, routed["mass"])
                weight = calc.request.terms[0].mask.float()
                calc.partials["halt_mass"] = (routed["mass"] * weight[..., None]).sum(
                    dim=(0, 1)
                )
                calc.partials["halt_count"] = weight.sum()
            else:
                loss = initial_ce
                calc.partials["halt_mass"] = loss.new_zeros(
                    self.max_reconsiderations + 1
                )
                calc.partials["halt_count"] = loss.new_zeros(())
            calc.add(0, loss)

        return runtime.predictions(
            hidden,
            {},
            positions,
            labels=labels,
            mask=mask,
            context_inputs=context_inputs,
            default_loss=self.default_loss,
            compute=compute,
        )

    def finalize(self, result, *, synchronize):
        tokens = result.objective_result()
        mass, count = result.partials["halt_mass"], result.partials["halt_count"]
        counts = torch.stack((count, tokens.denominator))
        if (
            synchronize
            and dist.is_initialized()
            and dist.get_world_size(group=get_dp_group()) > 1
        ):
            # This runs once over all chunks. The trainer scales the accumulated
            # token sums by the global supervised denominator.
            mass = all_reduce(mass, group=get_dp_group())
            dist.all_reduce(counts, group=get_dp_group())
        probability = mass / counts[0].clamp_min(1)
        entropy = -(
            probability
            * probability.clamp_min(torch.finfo(probability.dtype).tiny).log()
        ).sum()
        # A reduced draft vocabulary can exclude every teacher token. Keep
        # that minibatch's loss zero and its native optimizer denominator valid.
        normalizer = tokens.denominator.where(counts[1] > 0, 1)
        tokens.ratios["halt_entropy"] = (entropy.detach() * normalizer, normalizer)
        return ObjectiveResult(
            tokens.numerator - self.balance_beta * entropy * normalizer,
            normalizer,
            tokens.ratios,
        )
