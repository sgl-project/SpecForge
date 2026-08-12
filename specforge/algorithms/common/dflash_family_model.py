# coding=utf-8
"""DFlash-family training models and shared masking helpers."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.core.chunking import checkpointed_chunk_reduce
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.flex_attention_backend import flex_attention_backend
from specforge.modeling.draft.vocab_mixin import OUT_OF_DRAFT_VOCAB_LABEL

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None
    create_block_mask = None

# NPU workaround: flex_attention is not available on Ascend NPU.
if hasattr(torch, "npu") and torch.npu.is_available():
    FLEX_ATTENTION_AVAILABLE = False


#: Vocabulary rows per chunk when reducing the teacher's full-vocab normalizer.
#: Bounds a transient [tokens, chunk] tensor only; it does not affect results.
_TEACHER_NORMALIZER_VOCAB_CHUNK = 32768

_VALID_LOSS_TYPES = {
    "dflash",
    "dpace",
    "dpace-cumulative-confidence-only",
    "dpace-continuation-value-only",
}
_DPACE_LOSS_TYPES = _VALID_LOSS_TYPES - {"dflash"}


def compute_accept_len(
    pred_ids_4d: torch.Tensor,
    target_ids_4d: torch.Tensor,
    valid_mask_4d: torch.Tensor,
) -> torch.Tensor:
    """Compute per-block acceptance length."""
    correct = (pred_ids_4d == target_ids_4d) | (~valid_mask_4d)
    accept_prefix = correct.long().cumprod(dim=2) * valid_mask_4d.long()
    return accept_prefix.sum(dim=2).float()


def create_dflash_sdpa_mask(
    anchor_positions,
    block_keep_mask,
    S,
    block_size,
    device,
    sliding_window: Optional[int] = None,
):
    """Construct a full or sliding dense boolean DFlash mask."""

    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("sliding_window must be > 0")
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    q_indices = torch.arange(Q_LEN, device=device).view(1, 1, -1, 1)  # (1, 1, Q_LEN, 1)
    kv_indices = torch.arange(KV_LEN, device=device).view(
        1, 1, 1, -1
    )  # (1, 1, 1, KV_LEN)

    q_block_ids = q_indices // block_size
    q_block_offsets = q_indices % block_size

    anchor_expanded = anchor_positions.view(B, 1, N, 1).repeat_interleave(
        block_size, dim=2
    )

    mask_context = (kv_indices < S) & (kv_indices < anchor_expanded)
    if sliding_window is not None:
        # The current draft token occupies one slot in the window.
        context_lower_bound = anchor_expanded + q_block_offsets - (sliding_window - 1)
        mask_context = mask_context & (kv_indices >= context_lower_bound)

    is_draft = kv_indices >= S
    kv_block_ids = (kv_indices - S) // block_size
    mask_draft = is_draft & (q_block_ids == kv_block_ids)
    if sliding_window is not None:
        kv_block_offsets = (kv_indices - S) % block_size
        mask_draft = mask_draft & (kv_block_offsets <= q_block_offsets)

    valid_block = block_keep_mask.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)

    final_mask = (mask_context | mask_draft) & valid_block
    return final_mask


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
    flex_block_size=None,
    sliding_window: Optional[int] = None,
):
    """Construct a full or sliding Flex Attention mask for DFlash training."""

    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("sliding_window must be > 0")

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        q_block_offset = q_idx % block_size
        safe_q_block_id = q_block_id.clamp(max=N - 1)
        anchor_pos = anchor_positions[b, safe_q_block_id]

        is_context = kv_idx < S
        # Strictly less than: matches inference where target_hidden[anchor_pos]
        # is not available as context.
        mask_context = is_context & (kv_idx < anchor_pos)
        if sliding_window is not None:
            # The current draft token occupies one slot in the window.
            context_lower_bound = anchor_pos + q_block_offset - (sliding_window - 1)
            mask_context = mask_context & (kv_idx >= context_lower_bound)

        is_draft = kv_idx >= S
        kv_block_id = (kv_idx - S) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)
        if sliding_window is not None:
            kv_block_offset = (kv_idx - S) % block_size
            mask_draft = mask_draft & (kv_block_offset <= q_block_offset)

        is_valid_block = block_keep_mask[b, safe_q_block_id]
        in_bounds = q_block_id < N
        return (mask_context | mask_draft) & is_valid_block & in_bounds

    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    kwargs = {}
    if flex_block_size is not None:
        kwargs["BLOCK_SIZE"] = flex_block_size
    return create_block_mask(
        dflash_mask_mod,
        B=B,
        H=None,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        device=device,
        **kwargs,
    )


class OnlineDFlashModel(nn.Module):
    """DFlash online training wrapper with DFlash and D-PACE losses."""

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        objective_chunk_blocks: int = 128,
        loss_type: str = "dflash",
        dpace_alpha: float = 0.5,
    ):
        super().__init__()
        if loss_type not in _VALID_LOSS_TYPES:
            raise ValueError(
                f"loss_type={loss_type!r}; must be one of {sorted(_VALID_LOSS_TYPES)}"
            )
        if not 0.0 <= dpace_alpha <= 1.0:
            raise ValueError(f"dpace_alpha must be in [0, 1], got {dpace_alpha}")
        if objective_chunk_blocks < 0:
            raise ValueError("objective_chunk_blocks must be >= 0")

        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.use_draft_vocab = bool(getattr(draft_model, "use_draft_vocab", False))
        # (version, pruned_head_weight, target->draft label index); see
        # _pruned_head_state for why this cannot be built in __init__.
        self._pruned_head_cache: Optional[Tuple[int, torch.Tensor, torch.Tensor]] = None
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma
        self.objective_chunk_blocks = int(objective_chunk_blocks)
        self.loss_type = loss_type
        self.dpace_alpha = dpace_alpha

        self._cached_block_mask: Optional[BlockMask] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_bsz: Optional[int] = None

    def _pruned_head_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(head_weight[K, H], label_index[V])`` for the pruned vocab.

        Built on first use rather than in ``__init__``: the training model is
        constructed in ``build_model_bundle`` but the t2d/d2t mapping is only
        installed afterwards (``_ensure_offline_vocab_mapping``, or an explicit
        ``model.vocab_mapping_path``). Keying the cache on the draft model's
        ``vocab_mapping_version`` makes a later re-install take effect instead of
        silently serving a stale slice.
        """
        draft = self.draft_model
        version = int(getattr(draft, "vocab_mapping_version", 0))
        cache = self._pruned_head_cache
        if cache is not None and cache[0] == version:
            return cache[1], cache[2]
        draft.require_vocab_mapping()
        weight = self.lm_head.weight
        mask = draft.t2d.to(device=weight.device, dtype=torch.bool)
        # Row-slicing keeps the same order as column-slicing the full logits, so
        # draft index i corresponds to target id nonzero(t2d)[i] on both sides.
        pruned_weight = weight[mask].contiguous()
        label_index = draft.draft_vocab_index().to(device=weight.device)
        self._pruned_head_cache = (version, pruned_weight, label_index)
        return pruned_weight, label_index

    def apply_objective_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states onto the vocabulary this objective supervises.

        Unpruned runs call the head module itself, which keeps any head that is
        not a plain ``nn.Linear`` working. Only pruning needs the raw rows.
        """
        if not self.use_draft_vocab:
            return self.lm_head(hidden_states)
        weight = self._pruned_head_state()[0]
        return F.linear(hidden_states.to(weight.dtype), weight)

    def _full_vocab_logsumexp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``log sum_v exp(target_logit_v)`` over the FULL target vocabulary.

        Needed to turn pruned logits into true probabilities instead of
        probabilities conditioned on the token having survived pruning. Only the
        scalar normalizer is wanted, so the vocabulary is walked in chunks and
        each chunk is reduced immediately: the ``[..., vocab_size]`` tensor that
        pruning exists to avoid is never materialized. Callers hold
        ``torch.no_grad()``, so this costs one extra projection in compute and
        nothing in activation memory.

        Returns ``hidden_states`` without its feature dimension.
        """
        weight = self.lm_head.weight
        partials = []
        for start in range(0, int(weight.shape[0]), _TEACHER_NORMALIZER_VOCAB_CHUNK):
            rows = weight[start : start + _TEACHER_NORMALIZER_VOCAB_CHUNK]
            chunk_logits = F.linear(hidden_states.to(rows.dtype), rows).float()
            partials.append(torch.logsumexp(chunk_logits, dim=-1))
        # logsumexp is associative over any partition of the vocabulary.
        return torch.logsumexp(torch.stack(partials, dim=-1), dim=-1)

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample anchors whose clean token and first target are supervised."""

        num_candidates = max(seq_len - 1, 0)
        valid = (loss_mask[:, :num_candidates] > 0.5) & (
            loss_mask[:, 1 : num_candidates + 1] > 0.5
        )
        valid_counts = valid.sum(dim=1)
        width = min(self.num_anchors, int(valid_counts.max().item()))
        if width == 0:
            raise ValueError(
                "DFlash-family training requires two consecutive supervised tokens"
            )

        random_values = torch.rand(valid.shape, device=device)
        random_values.masked_fill_(~valid, 2.0)
        candidates = random_values.argsort(dim=1)[:, :width]
        keep_mask = torch.arange(width, device=device).unsqueeze(
            0
        ) < valid_counts.clamp(max=width).unsqueeze(1)

        sentinel = valid.shape[1]
        anchors = torch.where(
            keep_mask,
            candidates,
            torch.full_like(candidates, sentinel),
        )
        anchors = anchors.sort(dim=1).values
        keep_mask = anchors < sentinel
        return torch.where(keep_mask, anchors, 0), keep_mask

    def _create_position_ids(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        """Create absolute position IDs for parallel draft blocks."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        return pos_ids.view(bsz, -1)

    def _create_noise_embed(self, input_ids, anchor_positions, block_keep_mask):
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * bs), self.mask_token_id, dtype=torch.long, device=device
        )

        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )

        return self.embed_tokens(noise_ids)

    def _dpace_weight(
        self,
        prob: torch.Tensor,
        binary_mask: torch.Tensor,
        binary_mask_b: torch.Tensor,
        loss_type: str,
    ) -> torch.Tensor:
        """Compute detached D-PACE position weights.

        ``prob`` is the draft probability on the target token at each draft
        position. Invalid positions are treated as multiplicative no-ops inside
        prefix products and excluded from suffix sums; the caller still
        multiplies the returned weights by ``binary_mask`` before reduction.
        """
        smooth = (1.0 - self.dpace_alpha) * prob + self.dpace_alpha
        smooth = torch.where(binary_mask_b, smooth, torch.ones_like(smooth))
        prefix = torch.cumprod(smooth, dim=-1)

        if loss_type == "dpace-cumulative-confidence-only":
            return prefix

        suffix = torch.flip(
            torch.cumsum(torch.flip(prefix * binary_mask, dims=[-1]), dim=-1),
            dims=[-1],
        )

        if loss_type == "dpace":
            return suffix
        if loss_type == "dpace-continuation-value-only":
            return suffix / prefix.clamp_min(torch.finfo(prefix.dtype).tiny)
        raise ValueError(f"unknown D-PACE loss_type {loss_type!r}")

    def _forward_draft_blocks(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )

        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )

        context_position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        )
        draft_position_ids = self._create_position_ids(anchor_positions)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        mask_builder = (
            create_dflash_block_mask
            if self.attention_backend == "flex_attention"
            else create_dflash_sdpa_mask
        )
        mask_args = {
            "anchor_positions": anchor_positions,
            "block_keep_mask": block_keep_mask,
            "S": seq_len,
            "block_size": self.block_size,
            "device": device,
        }
        if (
            self.attention_backend == "flex_attention"
            and flex_attention_backend() == "FLASH"
        ):
            # FLASH requires a minimum of this block size.
            mask_args["flex_block_size"] = (256, 128)
        full_attn_mask = mask_builder(**mask_args)
        sliding_window = self.draft_model.sliding_window
        dflash_attn_mask = full_attn_mask
        if sliding_window is not None:
            dflash_attn_mask = {
                "full_attention": full_attn_mask,
                "sliding_attention": mask_builder(
                    **mask_args,
                    sliding_window=sliding_window,
                ),
            }

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attn_mask,
        )
        return anchor_positions, block_keep_mask, output_hidden

    def _dflash_objective_chunk_terms(
        self,
        hidden: torch.Tensor,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Return additive DFlash/D-PACE loss and accuracy terms."""

        batch_size, num_blocks, block_size, hidden_size = hidden.shape
        logits = self.lm_head(
            hidden.reshape(batch_size, num_blocks * block_size, hidden_size)
        ).reshape(batch_size, num_blocks, block_size, -1)
        neg_log_q = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape_as(target_ids)

        if self.loss_type == "dflash":
            loss_weights = weight_mask
            if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
                positions = torch.arange(
                    self.block_size,
                    device=hidden.device,
                ).view(1, 1, -1)
                decay_weights = torch.exp(
                    -(positions - 1).clamp(min=0).float() / self.loss_decay_gamma
                )
                loss_weights = loss_weights * decay_weights
            loss_num = (neg_log_q * loss_weights).sum()
            loss_den = loss_weights.sum()
        elif self.loss_type in _DPACE_LOSS_TYPES:
            with torch.no_grad():
                target_probability = torch.exp(-neg_log_q)
                dpace_weights = self._dpace_weight(
                    target_probability,
                    weight_mask,
                    weight_mask > 0,
                    self.loss_type,
                )
            loss_num = (neg_log_q * weight_mask * dpace_weights).sum()
            loss_den = loss_num.new_zeros(())
        else:  # defensive: __init__ validates the configured loss type.
            raise ValueError(f"unknown loss_type {self.loss_type!r}")

        with torch.no_grad():
            predicted_ids = logits.argmax(dim=-1)
            correct_num = (
                ((predicted_ids == target_ids) & (weight_mask > 0.5)).sum().float()
            )
            accuracy_den = weight_mask.sum()
        return loss_num, loss_den, correct_num, accuracy_den

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
        """Parallel block-wise training forward pass; returns
        (loss, accuracy, metrics) — same shape as Domino's forward."""
        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask, output_hidden = self._forward_draft_blocks(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
        )

        # --- Labels: same-position prediction (position k predicts token anchor+k) ---
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        # --- Weight mask: block validity * bounds * exclude anchor (pos 0) * loss_mask ---
        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()

        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        hidden_4d = output_hidden.reshape(
            bsz,
            anchor_positions.shape[1],
            self.block_size,
            -1,
        )
        loss_num, loss_den, correct_num, accuracy_denom = checkpointed_chunk_reduce(
            self._dflash_objective_chunk_terms,
            hidden_4d,
            target_ids,
            weight_mask,
            chunk_size=self.objective_chunk_blocks,
            dim=1,
        )
        ratio_metrics = {
            "acc": (correct_num.detach(), accuracy_denom.detach()),
        }
        metrics: Dict[str, object] = {
            "accuracy_denom": accuracy_denom.detach(),
            "ratio_metrics": ratio_metrics,
        }
        loss_denominator = (
            loss_den if self.loss_type == "dflash" else loss_num.new_tensor(float(bsz))
        )
        loss = loss_num / loss_denominator
        metrics["loss_terms"] = (loss_num, loss_denominator.detach())
        accuracy = correct_num / accuracy_denom
        return loss, accuracy, metrics


class OnlineDominoModel(OnlineDFlashModel):
    """Domino online training wrapper over DFlash block-parallel components."""

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        objective_chunk_blocks: int = 128,
        shift_label: bool = False,
    ):
        super().__init__(
            draft_model=draft_model,
            target_lm_head=target_lm_head,
            target_embed_tokens=target_embed_tokens,
            mask_token_id=mask_token_id,
            block_size=block_size,
            attention_backend=attention_backend,
            num_anchors=num_anchors,
            loss_decay_gamma=loss_decay_gamma,
            objective_chunk_blocks=objective_chunk_blocks,
            loss_type="dflash",
        )
        self.shift_label = shift_label

    def _build_domino_head_inputs(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        target_ids: torch.Tensor,
        output_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, n, bs = target_ids.shape
        hidden4d = output_hidden.reshape(bsz, n, bs, output_hidden.shape[-1])

        prev_ids = target_ids
        if self.shift_label:
            prev_offsets = torch.arange(
                0, self.block_size, device=input_ids.device
            ).view(1, 1, -1)
            prev_indices = (anchor_positions.unsqueeze(-1) + prev_offsets).clamp(
                max=input_ids.size(1) - 1
            )
            prev_ids = torch.gather(
                input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
                2,
                prev_indices,
            )

        return hidden4d, prev_ids

    def _apply_domino_head(
        self,
        base_logits4d: torch.Tensor,
        hidden4d: torch.Tensor,
        prev_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        head_token_ids = prev_ids if self.shift_label else target_ids
        head_token_embeddings = self.embed_tokens(head_token_ids)
        return self.draft_model.apply_logits_head(
            base_logits4d,
            hidden_states=hidden4d,
            prev_token_embeddings=head_token_embeddings,
        )

    def _domino_objective_chunk_terms(
        self,
        hidden: torch.Tensor,
        prev_ids: torch.Tensor,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
        eval_weight_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Return additive Domino loss and telemetry terms for one block slice."""

        batch_size, num_blocks, block_size, hidden_size = hidden.shape
        base_logits = self.lm_head(
            hidden.reshape(batch_size, num_blocks * block_size, hidden_size)
        ).reshape(batch_size, num_blocks, block_size, -1)
        final_logits = self._apply_domino_head(
            base_logits4d=base_logits,
            hidden4d=hidden,
            prev_ids=prev_ids,
            target_ids=target_ids,
        )
        final_ce = F.cross_entropy(
            final_logits.reshape(-1, final_logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape_as(target_ids)
        base_ce = F.cross_entropy(
            base_logits.reshape(-1, base_logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape_as(target_ids)
        final_num = (final_ce * weight_mask).sum()
        base_num = (base_ce * weight_mask).sum()
        loss_den = weight_mask.sum()

        with torch.no_grad():
            predicted_ids = final_logits.argmax(dim=-1)
            base_predicted_ids = base_logits.argmax(dim=-1)
            binary_accuracy_mask = eval_weight_mask > 0.5
            correct_num = (
                ((predicted_ids == target_ids) & binary_accuracy_mask).sum().float()
            )
            base_correct_num = (
                ((base_predicted_ids == target_ids) & binary_accuracy_mask)
                .sum()
                .float()
            )
            accuracy_den = eval_weight_mask.sum()

            valid_mask = eval_weight_mask > 0
            accepted = compute_accept_len(predicted_ids, target_ids, valid_mask)
            base_accepted = compute_accept_len(
                base_predicted_ids,
                target_ids,
                valid_mask,
            )
            valid_blocks = valid_mask.any(dim=-1).float()
            accept_num = ((accepted + 1.0) * valid_blocks).sum()
            base_accept_num = ((base_accepted + 1.0) * valid_blocks).sum()
            accept_den = valid_blocks.sum()

        return (
            final_num,
            base_num,
            loss_den,
            correct_num,
            base_correct_num,
            accuracy_den,
            accept_num,
            base_accept_num,
            accept_den,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        lambda_base: float = 0.0,
    ):
        """Parallel Domino training forward pass."""
        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask, output_hidden = self._forward_draft_blocks(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
        )

        label_start = 1 if self.shift_label else 0
        label_offsets = torch.arange(
            label_start, label_start + self.block_size, device=device
        ).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_target_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_target_indices,
        )

        bsz, n, bs = target_ids.shape
        hidden4d, prev_ids = self._build_domino_head_inputs(
            input_ids=input_ids,
            anchor_positions=anchor_positions,
            target_ids=target_ids,
            output_hidden=output_hidden,
        )
        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()

        if not self.shift_label:
            pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
            weight_mask = weight_mask * (pos_in_block > 0).float()

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_target_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        eval_weight_mask = weight_mask.clone()

        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            offset = 0 if self.shift_label else 1
            decay_weights = torch.exp(
                -(k - offset).clamp(min=0).float() / self.loss_decay_gamma
            )
            weight_mask = weight_mask * decay_weights

        (
            final_num,
            base_num,
            loss_den,
            correct_num,
            base_correct_num,
            accuracy_denom,
            accept_num,
            base_accept_num,
            accept_den,
        ) = checkpointed_chunk_reduce(
            self._domino_objective_chunk_terms,
            hidden4d,
            prev_ids,
            target_ids,
            weight_mask,
            eval_weight_mask,
            chunk_size=self.objective_chunk_blocks,
            dim=1,
        )

        valid_token_count = loss_den + 1e-6
        final_loss = final_num / valid_token_count
        base_loss = base_num / valid_token_count
        loss = (1.0 - lambda_base) * final_loss + lambda_base * base_loss
        accuracy = correct_num / (accuracy_denom + 1e-6)
        metrics = {
            "final_loss": final_loss.detach(),
            "base_loss": base_loss.detach(),
            "base_accuracy": (base_correct_num / (accuracy_denom + 1e-6)).detach(),
            "accept_len": (accept_num / (accept_den + 1e-6)).detach(),
            "base_accept_len": (base_accept_num / (accept_den + 1e-6)).detach(),
            "lambda_base": torch.tensor(lambda_base, device=loss.device),
            "accuracy_denom": accuracy_denom.detach(),
        }

        return loss, accuracy, metrics


class OnlineDSparkModel(OnlineDFlashModel):
    """DSpark online training wrapper over DFlash block-parallel components."""

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        dspark_ce_loss_alpha: float = 0.1,
        dspark_l1_loss_alpha: float = 0.9,
        dspark_confidence_head_alpha: float = 1.0,
        objective_chunk_blocks: int = 128,
    ):
        super().__init__(
            draft_model=draft_model,
            target_lm_head=target_lm_head,
            target_embed_tokens=target_embed_tokens,
            mask_token_id=mask_token_id,
            block_size=block_size,
            attention_backend=attention_backend,
            num_anchors=num_anchors,
            loss_decay_gamma=loss_decay_gamma,
            objective_chunk_blocks=objective_chunk_blocks,
            loss_type="dflash",
        )
        if dspark_ce_loss_alpha < 0:
            raise ValueError("dspark_ce_loss_alpha must be >= 0")
        if dspark_l1_loss_alpha < 0:
            raise ValueError("dspark_l1_loss_alpha must be >= 0")
        if dspark_confidence_head_alpha < 0:
            raise ValueError("dspark_confidence_head_alpha must be >= 0")
        self.loss_type = "dspark"
        self.dspark_ce_loss_alpha = float(dspark_ce_loss_alpha)
        self.dspark_l1_loss_alpha = float(dspark_l1_loss_alpha)
        self.dspark_confidence_head_alpha = float(dspark_confidence_head_alpha)

    def _build_dspark_labels_and_mask(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = input_ids.shape[1]
        device = input_ids.device
        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(
            1, 1, -1
        )
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        target_valid = label_indices < seq_len
        target_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, label_indices.size(1), -1),
            2,
            safe_label_indices,
        )
        eval_mask = target_valid & (target_loss_mask > 0.5)
        eval_mask = eval_mask & block_keep_mask.unsqueeze(-1)
        eval_mask = eval_mask.to(torch.int32).cumprod(dim=-1).bool()
        return target_ids, eval_mask, safe_label_indices

    def _dspark_loss_weight_mask(
        self,
        eval_mask: torch.Tensor,
    ) -> torch.Tensor:
        loss_weight_mask = eval_mask.to(torch.float32)
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            positions = torch.arange(self.block_size, device=eval_mask.device).view(
                1, 1, -1
            )
            decay_weights = torch.exp(-positions.float() / float(self.loss_decay_gamma))
            loss_weight_mask = loss_weight_mask * decay_weights
        return loss_weight_mask

    def _aligned_target_hidden(
        self,
        target_last_hidden_states: torch.Tensor,
        safe_label_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Gather the target state that predicts each DSpark label token."""

        target_pred_indices = (safe_label_indices - 1).clamp(min=0)
        batch_size = target_last_hidden_states.shape[0]
        hidden_size = target_last_hidden_states.shape[-1]
        gather_indices = target_pred_indices.reshape(batch_size, -1, 1).expand(
            -1, -1, hidden_size
        )
        return torch.gather(
            target_last_hidden_states,
            1,
            gather_indices,
        ).reshape(*safe_label_indices.shape, hidden_size)

    def _dspark_objective_chunk_terms(
        self,
        hidden: torch.Tensor,
        prev_token_ids: torch.Tensor,
        target_ids: torch.Tensor,
        loss_weights: torch.Tensor,
        ce_weights: torch.Tensor,
        eval_mask: torch.Tensor,
        aligned_target_hidden: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        """Return additive loss and telemetry numerators for one block slice.

        ``target_ids`` are labels in the objective's own vocabulary: real target
        ids for a full-vocabulary draft, draft ids (with
        ``OUT_OF_DRAFT_VOCAB_LABEL`` for pruned tokens) when the vocab is pruned.
        ``prev_token_ids`` are always real target ids -- they index the Markov
        head's W1, which spans the full target vocabulary either way.
        """

        batch_size, num_blocks, block_size, hidden_size = hidden.shape
        base_logits = self.apply_objective_head(
            hidden.reshape(batch_size, num_blocks * block_size, hidden_size)
        ).reshape(batch_size, num_blocks, block_size, -1)
        draft_logits = self.draft_model.apply_logits_head(
            base_logits,
            prev_token_ids=prev_token_ids,
            hidden_states=hidden,
        )
        vocab_size = draft_logits.shape[-1]
        # ignore_index zeroes the pruned-label positions; ce_weights zeroes the
        # matching entries so they leave the CE denominator too.
        cross_entropy = F.cross_entropy(
            draft_logits.reshape(-1, vocab_size),
            target_ids.reshape(-1),
            reduction="none",
            ignore_index=OUT_OF_DRAFT_VOCAB_LABEL,
        ).reshape_as(target_ids)
        ce_num = (cross_entropy * ce_weights).sum()

        zero = ce_num.new_zeros(())
        l1_num = zero
        confidence_num = zero
        confidence_error_num = zero
        teacher_agreement_num = zero
        teacher_top1_num = zero
        draft_top1_num = zero
        tau_num = zero
        tau_den = zero
        accept_probability = None

        draft_probabilities = None
        teacher_ids = None
        kept_mass_num = zero
        if aligned_target_hidden is not None:
            flat_teacher = aligned_target_hidden.reshape(
                batch_size,
                num_blocks * block_size,
                hidden_size,
            )
            with torch.no_grad():
                # The teacher must use the SAME head rows as the student: that is
                # what turns the pruned objective into a match against the target
                # distribution renormalized over the kept tokens, rather than a
                # match against a truncated (sub-normalized) one.
                target_logits = self.apply_objective_head(flat_teacher).reshape_as(
                    draft_logits
                )
                # Distillation target: renormalized over the kept tokens. The
                # draft cannot place mass outside them, so its distribution sums
                # to one there; a target summing to the kept mass instead would
                # give the loss an irreducible floor and bias every gradient.
                teacher_conditional = torch.softmax(target_logits.float(), dim=-1)
                teacher_ids = target_logits.argmax(dim=-1)
                if self.use_draft_vocab:
                    # Acceptance target: the target's TRUE probabilities, which
                    # sum to the kept mass, not to one. Renormalizing here would
                    # claim a token is accepted at the rate it would have if the
                    # pruned tokens did not exist -- see _full_vocab_logsumexp.
                    full_log_normalizer = self._full_vocab_logsumexp(
                        flat_teacher
                    ).reshape(batch_size, num_blocks, block_size, 1)
                    target_probabilities = torch.exp(
                        target_logits.float() - full_log_normalizer
                    )
                else:
                    target_probabilities = teacher_conditional
            draft_probabilities = torch.softmax(draft_logits.float(), dim=-1)
            l1_per_token = (draft_probabilities - teacher_conditional).abs().sum(dim=-1)
            if self.use_draft_vocab:
                # sum_i min(q_i, p_i) over the kept tokens: the exact single-step
                # acceptance rate for a draft that can only propose them. Bounded
                # above by the kept mass, which is the ceiling pruning imposes.
                accept_probability = (
                    torch.minimum(draft_probabilities, target_probabilities)
                    .sum(dim=-1)
                    .clamp(0.0, 1.0)
                )
                kept_mass_num = (target_probabilities.sum(dim=-1) * eval_mask).sum()
            else:
                # Algebraically the same quantity when the teacher sums to one.
                # Kept as the original expression so the unpruned path stays
                # bit-identical rather than merely equivalent.
                accept_probability = (1.0 - 0.5 * l1_per_token).clamp(0.0, 1.0)
            if self.dspark_l1_loss_alpha > 0:
                l1_num = (l1_per_token * loss_weights).sum()

        confidence_pred = self.draft_model.predict_confidence(
            hidden,
            prev_token_ids=prev_token_ids,
        )
        if confidence_pred is not None and self.dspark_confidence_head_alpha > 0:
            if accept_probability is None:
                raise ValueError(
                    "DSpark confidence loss requires target_last_hidden_states"
                )
            confidence_per_token = F.binary_cross_entropy_with_logits(
                confidence_pred.float(),
                accept_probability.detach(),
                reduction="none",
            )
            confidence_num = (confidence_per_token * loss_weights).sum()
            confidence_error_num = (
                (confidence_pred.float().sigmoid() - accept_probability).abs()
                * loss_weights
            ).sum()

        with torch.no_grad():
            predicted_ids = draft_logits.argmax(dim=-1)
            # A pruned label is OUT_OF_DRAFT_VOCAB_LABEL, which no predicted id
            # can equal, so it counts as a miss -- exactly what serving does with
            # a token the draft cannot propose. Accuracy therefore reports the
            # real acceptance ceiling instead of hiding the pruning loss.
            correct = ((predicted_ids == target_ids) & eval_mask).float()
            correct_num = correct.sum()
            eval_den = eval_mask.float().sum()
            # Cross-entropy is undefined at pruned labels, so its telemetry uses
            # the in-vocabulary subset; accuracy keeps the full eval set.
            ce_eval_mask = eval_mask & (target_ids >= 0)
            ce_eval_den = ce_eval_mask.float().sum()
            ce_position_num = (cross_entropy.detach() * ce_eval_mask).sum(dim=(0, 1))
            ce_position_den = ce_eval_mask.float().sum(dim=(0, 1))
            correct_position_num = correct.sum(dim=(0, 1))
            position_den = eval_mask.float().sum(dim=(0, 1))
            if aligned_target_hidden is not None:
                assert draft_probabilities is not None and teacher_ids is not None
                teacher_agreement_num = (
                    (predicted_ids == teacher_ids).float() * eval_mask
                ).sum()
                teacher_top1_num = (
                    target_probabilities.max(dim=-1).values * eval_mask
                ).sum()
                draft_top1_num = (
                    draft_probabilities.max(dim=-1).values * eval_mask
                ).sum()
                valid_blocks = eval_mask.any(dim=-1).float()
                accepted_expectation = (
                    accept_probability.detach() * eval_mask
                ).cumprod(dim=-1).sum(dim=-1) + 1.0
                tau_num = (accepted_expectation * valid_blocks).sum()
                tau_den = valid_blocks.sum()

        return (
            ce_num,
            l1_num,
            confidence_num,
            confidence_error_num,
            correct_num,
            eval_den,
            ce_eval_den,
            ce_position_num,
            ce_position_den,
            correct_position_num,
            position_den,
            teacher_agreement_num,
            teacher_top1_num,
            draft_top1_num,
            tau_num,
            tau_den,
            kept_mass_num,
        )

    def _compute_dspark_loss(
        self,
        *,
        output_hidden: torch.Tensor,
        target_ids: torch.Tensor,
        eval_mask: torch.Tensor,
        prev_token_ids: torch.Tensor,
        safe_label_indices: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Token-pooled DSpark objective with bounded vocab-logit memory."""

        batch_size, num_blocks, block_size = target_ids.shape
        hidden_4d = output_hidden.reshape(
            batch_size,
            num_blocks,
            block_size,
            -1,
        )
        loss_weights = self._dspark_loss_weight_mask(eval_mask)
        local_loss_den = loss_weights.sum()
        # A pruned vocabulary splits the objective in two. The L1 and confidence
        # terms compare distributions and stay well defined at every supervised
        # position. Cross-entropy needs a realizable label, so it drops the
        # positions whose true token was pruned away -- and must drop them from
        # its denominator too, or alpha_ce would silently scale with coverage.
        # Unpruned, the two are the same tensor and the same sum, so reuse them
        # rather than recomputing: the point is to leave that path's operator
        # sequence exactly as it was, not merely to compute the same number.
        objective_target_ids = target_ids
        ce_weights = loss_weights
        local_ce_den = local_loss_den
        if self.use_draft_vocab:
            label_index = self._pruned_head_state()[1]
            objective_target_ids = label_index[target_ids]
            ce_weights = loss_weights * (objective_target_ids >= 0).to(
                loss_weights.dtype
            )
            local_ce_den = ce_weights.sum()
        need_target = self.dspark_l1_loss_alpha > 0 or (
            self.dspark_confidence_head_alpha > 0
            and getattr(self.draft_model, "confidence_head", None) is not None
        )
        aligned_target_hidden = None
        if need_target:
            if target_last_hidden_states is None:
                raise ValueError(
                    "DSpark L1/confidence loss requires target_last_hidden_states"
                )
            aligned_target_hidden = self._aligned_target_hidden(
                target_last_hidden_states,
                safe_label_indices,
            )

        totals = checkpointed_chunk_reduce(
            self._dspark_objective_chunk_terms,
            hidden_4d,
            prev_token_ids,
            objective_target_ids,
            loss_weights,
            ce_weights,
            eval_mask,
            aligned_target_hidden,
            chunk_size=self.objective_chunk_blocks,
            dim=1,
        )

        (
            ce_num,
            l1_num,
            confidence_num,
            confidence_error_num,
            correct_num,
            eval_den,
            ce_eval_den,
            ce_position_num,
            ce_position_den,
            correct_position_num,
            position_den,
            teacher_agreement_num,
            teacher_top1_num,
            draft_top1_num,
            tau_num,
            tau_den,
            kept_mass_num,
        ) = totals

        global_loss_den = local_loss_den.detach().clone()
        world_size = 1
        import torch.distributed as dist

        distributed = dist.is_available() and dist.is_initialized()
        if distributed:
            world_size = dist.get_world_size()
            if world_size > 1:
                dist.all_reduce(global_loss_den, op=dist.ReduceOp.SUM)
        if float(global_loss_den) <= 0:
            raise ValueError("DSpark objective has no supervised target tokens")

        # Unpruned, cross-entropy shares the objective's denominator exactly, so
        # it needs neither its own sum nor its own reduction. Emitting that
        # second collective unconditionally would change the collective sequence
        # of every existing run -- ranks must agree on the count and order of
        # collectives -- so this branch is part of leaving that path alone, not
        # a micro-optimization.
        global_ce_den = global_loss_den
        if self.use_draft_vocab:
            global_ce_den = local_ce_den.detach().clone()
            if world_size > 1:
                dist.all_reduce(global_ce_den, op=dist.ReduceOp.SUM)
            if self.dspark_ce_loss_alpha > 0 and float(global_ce_den) <= 0:
                raise ValueError(
                    "DSpark cross-entropy has no in-vocabulary target tokens; "
                    "the draft vocabulary covers none of this batch"
                )
        if not self.use_draft_vocab:
            # global_ce_den == global_loss_den here, so the split form below is
            # algebraically identical -- but not bit-identical: with a decayed
            # loss_weights the denominator is not an exactly representable value,
            # and x/D + y/D rounds differently from (x + y)/D. Keeping the
            # original expression means an existing run's loss curve does not
            # move by a few ulps for a feature it does not use.
            loss = (
                world_size
                * (
                    self.dspark_ce_loss_alpha * ce_num
                    + self.dspark_l1_loss_alpha * l1_num
                    + self.dspark_confidence_head_alpha * confidence_num
                )
                / global_loss_den
            )
        else:
            # Each term becomes a weighted mean over the positions it is defined
            # on, so alpha_ce keeps its meaning instead of being scaled by how
            # much of the batch the draft vocabulary happens to cover.
            ce_term = (
                self.dspark_ce_loss_alpha * ce_num / global_ce_den
                if self.dspark_ce_loss_alpha > 0
                else ce_num * 0.0
            )
            loss = world_size * (
                ce_term
                + (
                    self.dspark_l1_loss_alpha * l1_num
                    + self.dspark_confidence_head_alpha * confidence_num
                )
                / global_loss_den
            )

        ratio_metrics = {
            "acc": (correct_num, eval_den),
            "ce_loss": (ce_num.detach(), local_ce_den.detach()),
            "l1_loss": (l1_num.detach(), local_loss_den.detach()),
            "confidence_loss": (
                confidence_num.detach(),
                local_loss_den.detach(),
            ),
            "confidence_abs_error": (
                confidence_error_num.detach(),
                local_loss_den.detach(),
            ),
            "ce_position": (ce_position_num, ce_position_den),
            "accuracy_position": (correct_position_num, position_den),
        }
        if self.use_draft_vocab:
            # Share of supervised tokens the pruned vocabulary can actually
            # propose. This is the hard ceiling on acceptance, so it belongs in
            # the training log rather than in a post-hoc audit.
            ratio_metrics["draft_vocab_coverage"] = (ce_eval_den, eval_den)
        if aligned_target_hidden is not None:
            ratio_metrics.update(
                {
                    "teacher_agreement": (teacher_agreement_num, eval_den),
                    "teacher_top1_prob": (teacher_top1_num, eval_den),
                    "draft_top1_prob": (draft_top1_num, eval_den),
                    "tau_probabilistic": (tau_num, tau_den),
                }
            )
            if self.use_draft_vocab:
                # Target probability mass the pruned vocabulary can reach, in
                # probability rather than token counts. draft_vocab_coverage
                # answers "how often is the realized token proposable"; this
                # answers "how much of the teacher's belief survives pruning".
                # It upper-bounds each position's single-step acceptance
                # probability. It is NOT a bound on tau_probabilistic, which is
                # 1 + sum_j prod_{k<=j} accept_k -- an anchor token plus
                # cumulative products of those per-step probabilities.
                ratio_metrics["teacher_kept_mass"] = (kept_mass_num, eval_den)
        metrics: Dict[str, object] = {
            "ratio_metrics": {
                name: (numerator.detach(), denominator.detach())
                for name, (numerator, denominator) in ratio_metrics.items()
            },
            "accuracy_denom": eval_den.detach(),
        }
        accuracy = correct_num / eval_den.clamp_min(1.0)
        return loss, {"accuracy": accuracy.detach(), **metrics}

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
        """Parallel DSpark training forward pass."""
        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        anchor_positions, block_keep_mask, output_hidden = self._forward_draft_blocks(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
        )

        (
            target_ids,
            eval_mask,
            safe_label_indices,
        ) = self._build_dspark_labels_and_mask(
            input_ids=input_ids,
            loss_mask=loss_mask,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
        )
        anchor_token_ids = torch.gather(input_ids, 1, anchor_positions)
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]],
            dim=-1,
        )
        loss, metrics = self._compute_dspark_loss(
            output_hidden=output_hidden,
            target_ids=target_ids,
            eval_mask=eval_mask,
            prev_token_ids=prev_token_ids,
            safe_label_indices=safe_label_indices,
            target_last_hidden_states=target_last_hidden_states,
        )
        accuracy = metrics.pop("accuracy")
        return loss, accuracy, metrics
