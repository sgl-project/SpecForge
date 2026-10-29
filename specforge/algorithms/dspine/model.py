"""DSpine block training with frozen target distributions and token codes."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.core.chunking import checkpointed_chunk_reduce
from specforge.core.fused_loss import ce_l1_loss


class OnlineDSpineModel(OnlineDFlashModel):
    def __init__(self, **kwargs) -> None:
        if kwargs.get("loss_decay_gamma") is None:
            kwargs["loss_decay_gamma"] = 7.0
        super().__init__(**kwargs)
        self.loss_type = "dspine"
        self.lm_head.requires_grad_(False)
        self.embed_tokens.requires_grad_(False)
        if self.block_size != self.draft_model.block_size:
            raise ValueError("DSpine training block_size must match the draft")
        self.dspine_config = self.draft_model.dspine_config
        if not self.draft_model.transfer_ready.item():
            self.draft_model.initialize_transfer_space(self.lm_head.weight)

    def _ce_terms(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
        supervised: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        terms = self._logits_loss_terms(self.lm_head(hidden), labels)
        correct = ((terms.argmax_ids == labels) * supervised).sum()
        return (terms.cross_entropy * weights).sum(), correct.float()

    def _objective_terms(
        self,
        hidden: torch.Tensor,
        pre_injection: torch.Tensor,
        layer_features: torch.Tensor,
        labels: torch.Tensor,
        predecessors: torch.Tensor,
        teacher_hidden: torch.Tensor,
        weights: torch.Tensor,
        supervised: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        config = self.dspine_config
        with torch.no_grad():
            teacher_logits = self.target_logit_transform.apply_teacher(
                self.lm_head(teacher_hidden)
            )
        terms = self._logits_loss_terms(
            self.lm_head(hidden),
            labels,
            teacher_logits=teacher_logits,
            top_k=config.top_k,
        )
        candidates = terms.topk_ids
        refined = self.draft_model.refine(pre_injection, predecessors)
        scores = torch.einsum(
            "bmh,bmkh->bmk", refined, F.embedding(candidates, self.lm_head.weight)
        ).float()
        target_candidates = teacher_logits.softmax(dim=-1, dtype=torch.float32).gather(
            -1, candidates
        )
        # The gate uses full-vocabulary mass, before candidate normalization.
        mass = target_candidates.sum(dim=-1)
        covered = candidates == labels.unsqueeze(-1)
        # Normalize both candidate distributions through the same softmax path.
        refinement = ce_l1_loss(
            scores,
            covered.long().argmax(dim=-1),
            teacher_logits=teacher_logits.gather(-1, candidates),
        )
        # Missing labels use index zero above only as a placeholder for CE.
        refined_ce = refinement.cross_entropy * covered.any(dim=-1)
        refined_l1 = refinement.teacher_l1 * (mass > config.candidate_mass_threshold)
        reference_codes = F.embedding(labels, self.draft_model.transfer_codes).float()
        alignment = 1 - F.cosine_similarity(
            layer_features.float(), reference_codes.unsqueeze(1), dim=-1
        )
        with torch.no_grad():
            correct = ((terms.argmax_ids == labels) * supervised).sum()
            coverage = (covered.any(dim=-1) * supervised).sum()
        return (
            (terms.cross_entropy * weights).sum(),
            (terms.teacher_l1 * weights).sum(),
            (
                (config.ce_weight * refined_ce + (1 - config.ce_weight) * refined_l1)
                * weights
            ).sum(),
            (alignment * weights.unsqueeze(1)).sum(),
            correct.float(),
            coverage.float(),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: torch.Tensor,
        max_valid_anchors: int | None = None,
        global_step: int = 0,
        total_steps: int | None = None,
        collect_detailed_metrics: bool = True,
        alignment_ce: torch.Tensor | None = None,
        ce_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        del collect_detailed_metrics
        if ce_only and torch.is_grad_enabled():
            raise ValueError("DSpine CE prepass requires gradients to be disabled")
        batch, length = input_ids.shape
        anchors, keep = self._sample_anchor_positions(
            length, loss_mask, input_ids.device, max_valid_anchors
        )
        offsets = torch.arange(self.block_size, device=input_ids.device)
        indices = anchors.unsqueeze(-1) + offsets
        safe_indices = indices.clamp(max=length - 1)
        labels = (
            input_ids.unsqueeze(1)
            .expand(-1, anchors.shape[1], -1)
            .gather(2, safe_indices)
        )
        supervised = (indices < length) & keep.unsqueeze(-1)
        supervised = supervised & (
            loss_mask.unsqueeze(1)
            .expand(-1, anchors.shape[1], -1)
            .gather(2, safe_indices)
            > 0.5
        )
        supervised = supervised[..., 1:].int().cumprod(dim=-1).bool()
        weights = supervised.float()
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            weights = weights * torch.exp(-offsets[:-1].float() / self.loss_decay_gamma)
        denominator = weights.sum()
        beta, probability = self.dspine_config.schedule(global_step, total_steps)
        replace = None
        if self.training and probability > 0:
            replace = (
                torch.rand(anchors.shape, device=input_ids.device) < probability
            ) & keep
        position_ids = torch.cat(
            (
                torch.arange(length, device=input_ids.device).expand(batch, -1),
                self._create_position_ids(anchors),
            ),
            dim=1,
        )
        draft_kwargs = (
            {"kernel_options": {"BACKEND": "TRITON"}}
            if self.attention_backend == "flex_attention"
            else {}
        )
        output = self.draft_model(
            position_ids=position_ids,
            attention_mask=self._attention_mask(anchors, keep, length),
            noise_embedding=self._create_noise_embed(input_ids, anchors, keep),
            target_hidden=hidden_states.detach(),
            block_token_ids=labels,
            replace_mask=replace,
            **draft_kwargs,
        )
        tokens = self.block_size - 1
        if ce_only:
            ce, correct = checkpointed_chunk_reduce(
                self._ce_terms,
                output.hidden_states[..., 1:, :].reshape(
                    -1, tokens, output.hidden_states.shape[-1]
                ),
                labels[..., 1:].reshape(-1, tokens),
                weights.reshape(-1, tokens),
                supervised.reshape(-1, tokens),
                chunk_size=self.objective_chunk_blocks,
                checkpoint=self.objective_checkpointing,
            )
            count = supervised.sum().float()
            return (
                ce / denominator.clamp_min(1),
                correct / count.clamp_min(1),
                dict(
                    ratio_metrics={
                        "acc": (correct, count),
                        "dspine/backbone_ce": (ce, denominator),
                    },
                    accuracy_denom=count,
                    loss_terms=(ce, denominator),
                ),
            )
        target_indices = (safe_indices[..., 1:] - 1).clamp_min(0)
        target_hidden = target_last_hidden_states.detach().gather(
            1,
            target_indices.reshape(batch, -1, 1).expand(
                -1, -1, target_last_hidden_states.shape[-1]
            ),
        )
        layer_features = (
            output.layer_features[..., 1:, :].permute(1, 2, 0, 3, 4).flatten(0, 1)
        )
        ce, l1, refinement, alignment, correct, coverage = checkpointed_chunk_reduce(
            self._objective_terms,
            output.hidden_states[..., 1:, :].reshape(
                -1, tokens, output.hidden_states.shape[-1]
            ),
            output.pre_injection[..., 1:, :].reshape(
                -1, tokens, output.pre_injection.shape[-1]
            ),
            layer_features,
            labels[..., 1:].reshape(-1, tokens),
            labels[..., :-1].reshape(-1, tokens),
            target_hidden.reshape(-1, tokens, target_hidden.shape[-1]),
            weights.reshape(-1, tokens),
            supervised.reshape(-1, tokens),
            chunk_size=self.objective_chunk_blocks,
            checkpoint=self.objective_checkpointing,
        )
        if alignment_ce is None:
            scale_terms = torch.stack((ce.detach(), denominator.detach()))
            if dist.is_initialized():
                dist.all_reduce(scale_terms)
            torch._assert_async(
                scale_terms[1] > 0, "DSpine has no supervised proposal tokens"
            )
            alignment_ce = scale_terms[0] / scale_terms[1]
        alignment_scale = beta * alignment_ce.detach()
        numerator = (
            self.dspine_config.ce_weight * ce
            + (1 - self.dspine_config.ce_weight) * l1
            + refinement
            + alignment_scale * alignment
        )
        count = supervised.sum().float()
        ratios = {
            "acc": (correct, count),
            "dspine/backbone_ce": (ce.detach(), denominator),
            "dspine/backbone_l1": (l1.detach(), denominator),
            "dspine/refinement_loss": (refinement.detach(), denominator),
            "dspine/alignment_loss": (alignment.detach(), denominator),
            "dspine/candidate_coverage": (coverage, count),
        }
        metrics = dict(
            ratio_metrics=ratios,
            accuracy_denom=count,
            loss_terms=(numerator, denominator),
        )
        return (
            numerator / denominator.clamp_min(1),
            correct / count.clamp_min(1),
            metrics,
        )
