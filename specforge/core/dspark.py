# coding=utf-8
"""DSpark online training wrapper: DFlash backbone + Markov / L1 / confidence losses.

Ported from TorchSpec PR #129 (``torchspec/models/dspark.py``). Reuses SpecForge's
:class:`OnlineDFlashModel` anchor sampling, block-causal mask construction, and
MASK-token noise stream verbatim (via ``super()``), then layers on the DSpark
training objective:

  - Markov-biased draft logits (teacher-forced previous token).
  - Cross-entropy against the ground-truth next tokens (hard labels).
  - L1 distribution distillation: ``|softmax(draft) - softmax(target)|`` where the
    target distribution is the frozen LM head applied to the *target's* final
    hidden state at the aligned position (requires ``last_hidden_states``).
  - Confidence head BCE against the empirical per-token accept rate.

Combined: ``ce_alpha*ce + l1_alpha*l1 + confidence_alpha*confidence``.

Loss formulation adapted from DeepSeek's DeepSpec (``deepspec/modeling/dspark/loss.py``,
MIT), including its pooled global-mean reduction: local numerators over a
cross-rank all-reduced denominator, scaled by world_size to cancel FSDP's mean
gradient reduction.

Key SpecForge differences vs TorchSpec (see port notes in the PR):
  - SpecForge's :class:`OnlineDFlashModel.forward` returns ``(loss, accuracy)``;
    DSpark needs the per-component losses, so this forward returns a 6-tuple
    ``(loss, accuracy, loss_per_position, acc_per_position, count_per_position,
    loss_components)``. ``train_dspark.py`` consumes the extra elements.
  - The target ``lm_head`` is a frozen ``nn.Linear`` module on the wrapper
    (``self.lm_head``); the L1 path uses ``self.lm_head.weight`` for ``F.linear``.
  - The fused multi-layer context feature (``hidden_states``) is produced upstream
    by ``generate_dflash_data`` and fed straight to the draft as ``target_hidden``.
"""

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from specforge.core.dflash import (
    FLEX_ATTENTION_AVAILABLE,
    OnlineDFlashModel,
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
from specforge.modeling.draft.dspark import DSparkDraftModel


class OnlineDSparkModel(OnlineDFlashModel):
    """DSpark online training wrapper (DFlash backbone + Markov/L1/confidence heads)."""

    def __init__(
        self,
        draft_model: DSparkDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 7,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = 4.0,
        ce_loss_alpha: float = 0.1,
        l1_loss_alpha: float = 0.9,
        confidence_head_alpha: float = 1.0,
    ):
        # Reuse DFlash anchor/mask/noise machinery. loss_type="dflash" is only a
        # placeholder to satisfy the parent validator — DSpark overrides forward()
        # entirely and never dispatches on loss_type.
        super().__init__(
            draft_model=draft_model,
            target_lm_head=target_lm_head,
            target_embed_tokens=target_embed_tokens,
            mask_token_id=mask_token_id,
            block_size=block_size,
            attention_backend=attention_backend,
            num_anchors=num_anchors,
            loss_decay_gamma=loss_decay_gamma,
            loss_type="dflash",
        )
        self.ce_loss_alpha = float(ce_loss_alpha)
        self.l1_loss_alpha = float(l1_loss_alpha)
        self.confidence_head_alpha = float(confidence_head_alpha)

    def _decay_weights(self, device: torch.device) -> torch.Tensor:
        """exp(-k/gamma) over within-block position k (DeepSpec convention).

        Every slot 0..B-1 is a real prediction in DSpark (unlike DFlash, where
        slot 0 is the masked anchor), so slot 0 (the first predicted token) gets
        weight 1.0 and later slots decay.
        """
        k = torch.arange(self.block_size, device=device).view(1, 1, -1)
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            return torch.exp(-k.float() / self.loss_decay_gamma)
        return torch.ones_like(k, dtype=torch.float32)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
    ]:
        """DSpark training forward.

        ``hidden_states`` is the fused multi-layer context feature
        ``[B, S, len(target_layer_ids)*hidden]`` (the draft model applies its
        ``fc``/``hidden_norm`` internally). ``last_hidden_states`` is the target
        model's final hidden state ``[B, S, hidden]`` (needed only for the L1 /
        confidence objectives).

        Returns ``(loss, accuracy, loss_per_position, acc_per_position,
        count_per_position, loss_components)``. ``loss`` is the combined
        ce+l1+confidence objective; ``loss_components`` is a dict of detached
        per-rank local-mean scalars (ce_loss / l1_loss / confidence_loss) for
        logging.
        """
        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # ---- DFlash backbone (identical construction to OnlineDFlashModel.forward) ----
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        n_blocks = anchor_positions.shape[1]

        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )

        context_position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        )
        draft_position_ids = self._create_position_ids(anchor_positions)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        if self.attention_backend == "flex_attention":
            dflash_attn_mask = create_dflash_block_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
            )
        else:
            dflash_attn_mask = create_dflash_sdpa_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
            )

        draft_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attn_mask,
        )
        hidden_4d = draft_hidden.view(bsz, n_blocks, self.block_size, -1)

        base_logits = self.lm_head(draft_hidden)
        base_logits_4d = base_logits.view(bsz, n_blocks, self.block_size, -1)
        vocab_size = base_logits_4d.size(-1)

        # ---- Labels + eval mask (DSpark / DeepSpec convention) ----
        # Slot j predicts the token at anchor+j+1 (the real anchor token seeds
        # slot 0). All block_size slots are supervised — there is no masked anchor
        # slot, unlike SpecForge DFlash which drops slot 0.
        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(
            1, 1, -1
        )
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets  # [B, nb, bs]
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )  # [B, nb, bs]

        # eval mask = contiguous supervised prefix per block (DeepSpec
        # build_eval_mask): block kept, label in-bounds, target token supervised,
        # then cumprod so a gap truncates the rest of the block.
        target_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )
        eval_bool = (
            block_keep_mask.unsqueeze(-1) & valid_label_mask & (target_loss_mask > 0.5)
        )
        eval_bool = eval_bool.to(torch.int32).cumprod(dim=-1).bool()
        eval_mask = eval_bool.float()  # [B, nb, bs]

        decay_weight_mask = eval_mask * self._decay_weights(device)
        local_den = decay_weight_mask.sum()

        # ---- Markov-biased draft logits ----
        # prev token for slot j is the ground-truth token immediately before the
        # one slot j predicts: slot 0's prev is the real anchor token, slot j's is
        # target_ids[j-1]. Matches DeepSpec prev_token_ids.
        anchor_token_ids = torch.gather(input_ids, 1, anchor_positions)  # [B, nb]
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]], dim=-1
        )
        logits_4d = base_logits_4d
        if self.draft_model.markov_head is not None:
            logits_4d = self.draft_model.markov_head.apply_block_logits(
                base_logits_4d, token_ids=prev_token_ids
            )

        # ---- Cross entropy (hard labels) ----
        flat_logits = logits_4d.reshape(-1, vocab_size)
        flat_targets = target_ids.reshape(-1)
        ce_per_token = F.cross_entropy(
            flat_logits, flat_targets, reduction="none"
        ).view(bsz, n_blocks, self.block_size)
        ce_num = (ce_per_token * decay_weight_mask).sum()

        # ---- L1 distribution distillation + accept rate ----
        l1_num = base_logits.new_zeros((), dtype=torch.float32)
        accept_rate = None
        need_target = (self.l1_loss_alpha > 0) or (
            self.draft_model.confidence_head is not None
            and self.confidence_head_alpha > 0
        )
        if need_target:
            if last_hidden_states is None:
                raise ValueError(
                    "DSpark L1/confidence losses require target last_hidden_states; "
                    "ensure the target model surfaces its final hidden state."
                )
            # target distribution for the token at label_indices = target LM head
            # applied to the target hidden one position earlier (anchor+j).
            tgt_idx = (safe_label_indices - 1).clamp(min=0)  # [B, nb, bs]
            hdim = last_hidden_states.size(-1)
            gather_idx = tgt_idx.reshape(bsz, -1, 1).expand(-1, -1, hdim)
            aligned_hidden = torch.gather(last_hidden_states, 1, gather_idx)
            aligned_target_logits = F.linear(aligned_hidden, self.lm_head.weight).view(
                bsz, n_blocks, self.block_size, vocab_size
            )
            draft_probs = torch.softmax(logits_4d.float(), dim=-1)
            target_probs = torch.softmax(aligned_target_logits.float(), dim=-1)
            l1_per_token = (draft_probs - target_probs).abs().sum(dim=-1)  # [B, nb, bs]
            if self.l1_loss_alpha > 0:
                l1_num = (l1_per_token * decay_weight_mask).sum()
            accept_rate = (1.0 - 0.5 * l1_per_token).clamp(0.0, 1.0)

        # ---- Confidence head BCE ----
        conf_num = base_logits.new_zeros((), dtype=torch.float32)
        if (
            self.draft_model.confidence_head is not None
            and self.confidence_head_alpha > 0
        ):
            if self.draft_model.confidence_head_with_markov:
                prev_emb = self.draft_model.markov_head.get_prev_embeddings(
                    prev_token_ids
                ).to(hidden_4d.dtype)
                conf_features = torch.cat([hidden_4d, prev_emb], dim=-1)
            else:
                conf_features = hidden_4d
            confidence_pred = self.draft_model.confidence_head(conf_features).float()
            conf_bce = (
                F.binary_cross_entropy_with_logits(
                    confidence_pred, accept_rate.detach(), reduction="none"
                )
                * decay_weight_mask
            )
            conf_num = conf_bce.sum()

        # ---- Pooled global loss (DeepSpec _build_loss) ----
        # Local numerators over a cross-rank-summed denominator, x world_size to
        # cancel FSDP's mean gradient reduction -> a true token-pooled global mean
        # rather than a mean-of-per-rank-means.
        # NOTE: uses the global training group size; correct for plain DP / ZeRO-2
        # (single shard group). With a multi-dim mesh (e.g. HSDP/USP) the FSDP
        # shard group differs from world_size and this would need the shard group.
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        global_den = local_den.detach().clone()
        if world_size > 1:
            dist.all_reduce(global_den, op=dist.ReduceOp.SUM)
        global_den = global_den + 1e-6
        loss = (
            self.ce_loss_alpha * ce_num / global_den
            + self.l1_loss_alpha * l1_num / global_den
            + self.confidence_head_alpha * conf_num / global_den
        ) * world_size

        # Per-component loss values (per-rank local means) for logging only — lets
        # you watch L1 fall while the greedy-CE proxy plateaus.
        local_den_eps = local_den + 1e-6
        loss_components = {
            "ce_loss": (ce_num / local_den_eps).detach(),
            "l1_loss": (l1_num / local_den_eps).detach(),
            "confidence_loss": (conf_num / local_den_eps).detach(),
        }

        # ---- Metrics (cross-entropy based; all block_size slots are productive) ----
        with torch.no_grad():
            flat_binary = eval_mask.reshape(-1)
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (flat_binary > 0.5)
            accuracy = correct.sum().float() / flat_binary.sum().clamp(min=1e-6)

            count_per_position = eval_mask.sum(dim=(0, 1))
            count_pp = count_per_position.clamp(min=1.0)
            loss_per_position = (ce_per_token * eval_mask).sum(dim=(0, 1)) / count_pp
            acc_per_position = (
                correct.view(bsz, n_blocks, self.block_size).float().sum(dim=(0, 1))
                / count_pp
            )

        return (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            loss_components,
        )
