"""P-EAGLE (Parallel EAGLE) training wrapper with COD sampling."""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask

from specforge.core.loss import LogSoftmaxLoss
from specforge.modeling.draft.peagle import PEagleDraftModel


def generate_cod_sample_indices(
    seq_length: int,
    loss_mask: torch.Tensor,
    num_depths: int = 8,
    down_sample_ratio: float = 0.8,
    down_sample_ratio_min: float = 0.2,
    filter_position_zero: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loss_mask = loss_mask.squeeze(0)
    device = loss_mask.device
    all_valid_indices = torch.where(loss_mask == 1)[0]

    sample_indices = [torch.arange(seq_length, device=device)]
    n_per_depth = [seq_length]
    prev_indices = all_valid_indices

    for d in range(1, num_depths):
        valid_length = max(0, all_valid_indices.shape[0] - d)
        ratio = max(down_sample_ratio**d, down_sample_ratio_min)
        sample_size = int(valid_length * ratio)

        if sample_size <= 0:
            break

        if prev_indices.shape[0] >= sample_size:
            random_selection = torch.randperm(prev_indices.shape[0], device=device)[
                :sample_size
            ]
            sampled_idx = prev_indices[random_selection]
            sampled_idx = torch.sort(sampled_idx)[0]
        else:
            sampled_idx = prev_indices

        next_candidates = (sampled_idx + 1) % seq_length
        if filter_position_zero:
            next_candidates = next_candidates[next_candidates != 0]
        mask = torch.isin(next_candidates, all_valid_indices)
        prev_indices = next_candidates[mask]

        sample_indices.append(sampled_idx - d)
        n_per_depth.append(sampled_idx.shape[0])

    anchor_pos = torch.cat(sample_indices)
    depth = torch.cat(
        [
            torch.full((n,), i, device=device, dtype=torch.long)
            for i, n in enumerate(n_per_depth)
        ]
    )
    return anchor_pos, depth


def create_peagle_mask_mod(anchor_pos, depth, lengths, total_seq_len):
    document_ids = torch.repeat_interleave(
        torch.arange(lengths.shape[0], device=lengths.device, dtype=torch.long),
        lengths,
    )
    document_ids = torch.cat(
        [
            document_ids,
            -1
            * torch.ones(
                total_seq_len - document_ids.shape[0],
                device=lengths.device,
                dtype=torch.long,
            ),
        ]
    ).contiguous()

    def peagle_mask_mod(_b, _h, q_idx, kv_idx):
        q_anchor_pos = anchor_pos[q_idx]
        kv_anchor_pos = anchor_pos[kv_idx]
        q_depth = depth[q_idx]
        kv_depth = depth[kv_idx]

        same_document = document_ids[q_anchor_pos] == document_ids[kv_anchor_pos]
        is_not_padding = document_ids[q_anchor_pos] != -1
        same_rollout = q_anchor_pos == kv_anchor_pos
        kv_depth0 = kv_depth == 0
        in_depth_order = q_depth >= kv_depth
        is_anchor_causal = q_anchor_pos >= kv_anchor_pos

        return (
            is_not_padding
            & same_document
            & ((kv_depth0 & is_anchor_causal) | (same_rollout & in_depth_order))
        )

    return peagle_mask_mod


def compute_peagle_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    anchor_pos: torch.Tensor,
    depth: torch.Tensor,
    num_depths: int,
    t2d: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    device = logits.device
    orig_positions = anchor_pos + depth

    # Ensure loss_mask is 2D [batch, seq_len]
    if loss_mask.dim() == 3:
        loss_mask = loss_mask.squeeze(-1)

    sampled_loss_mask = loss_mask[:, orig_positions].float()  # [batch, total_sampled]

    # Map targets to draft vocabulary and skip positions whose target top-1
    # token is outside the draft vocabulary.
    target_logits = targets[:, orig_positions, :]
    if t2d is not None and t2d.dtype == torch.bool:
        target_top1 = targets.argmax(dim=-1)[:, orig_positions]
        target_in_draft_vocab = t2d[target_top1].to(sampled_loss_mask.dtype)
        sampled_loss_mask = sampled_loss_mask * target_in_draft_vocab
        target_logits = target_logits[:, :, t2d]

    target_p = torch.nn.functional.softmax(target_logits.float(), dim=-1)
    position_mask = sampled_loss_mask.unsqueeze(-1)  # [batch, total_sampled, 1]
    total_positions = position_mask.shape[0] * position_mask.shape[1]
    denominator = sampled_loss_mask.sum().clamp_min(1e-6)
    loss = LogSoftmaxLoss.apply(logits, target_p, position_mask) * (
        total_positions / denominator
    )

    with torch.no_grad():
        pred_ids = torch.argmax(logits, dim=-1)  # [batch, total_sampled]
        target_ids = torch.argmax(target_p, dim=-1)  # [batch, total_sampled]

        metrics: Dict[str, Any] = {
            "loss_sum": loss.detach(),
            "loss_total": torch.tensor(1.0, device=device),
        }

        correct_total = torch.tensor(0.0, device=device)
        count_total = torch.tensor(0.0, device=device)
        for d in range(num_depths):
            depth_mask = (depth == d).unsqueeze(0) & (sampled_loss_mask > 0.5)
            d_correct = ((pred_ids == target_ids) & depth_mask).sum().float()
            d_total = depth_mask.sum().float()
            metrics[f"position_{d}_acc_sum"] = d_correct
            metrics[f"position_{d}_acc_total"] = d_total
            correct_total += d_correct
            count_total += d_total

        metrics["full_acc_sum"] = correct_total
        metrics["full_acc_total"] = count_total

    return loss, metrics


class OnlinePEagleModel(nn.Module):

    def __init__(
        self,
        draft_model: PEagleDraftModel,
        mask_token_id: int,
        num_depths: int = 8,
        down_sample_ratio: float = 0.7,
        down_sample_ratio_min: float = 0.2,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.mask_token_id = mask_token_id
        self.num_depths = num_depths
        self.down_sample_ratio = down_sample_ratio
        self.down_sample_ratio_min = down_sample_ratio_min

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """P-EAGLE training forward pass.

        Args:
            input_ids: [batch, seq_len] - input token IDs
            attention_mask: [batch, seq_len] - padding mask
            target: [batch, seq_len, vocab_size] - target logits from target model
            loss_mask: [batch, seq_len] - which positions contribute to loss
            hidden_states: [batch, seq_len, 3*hidden_size] - concatenated aux hidden states
            lengths: [num_samples] - sequence lengths for multi-sample packing
        """
        device = hidden_states.device
        seq_length = input_ids.shape[1]

        # Ensure loss_mask is 2D [batch, seq_len]
        if loss_mask.dim() == 3:
            loss_mask = loss_mask.squeeze(-1)

        if lengths is None:
            lengths = torch.tensor([seq_length], dtype=torch.long, device=device)

        # Step 1: COD sampling
        anchor_pos, depth = generate_cod_sample_indices(
            seq_length=seq_length,
            loss_mask=loss_mask,
            num_depths=self.num_depths,
            down_sample_ratio=self.down_sample_ratio,
            down_sample_ratio_min=self.down_sample_ratio_min,
        )
        total_sampled = anchor_pos.shape[0]
        orig_positions = anchor_pos + depth
        is_depth_0 = depth == 0

        # Step 2: Build sampled input_ids
        sampled_ids = torch.where(
            is_depth_0,
            input_ids[0, orig_positions],
            torch.tensor(self.mask_token_id, dtype=input_ids.dtype, device=device),
        ).unsqueeze(0)

        inputs_embeds = self.draft_model.embed_input_ids(sampled_ids).to(
            hidden_states.dtype
        )

        # Step 3: Build sampled hidden states
        mask_hidden = self.draft_model.mask_hidden.to(
            device=device, dtype=hidden_states.dtype
        )
        sampled_hidden = torch.where(
            is_depth_0.unsqueeze(-1),
            hidden_states[0, orig_positions],
            mask_hidden.squeeze(0).expand(orig_positions.shape[0], -1),
        ).unsqueeze(0)

        # Step 4: Project and concatenate
        sampled_hidden = self.draft_model.project_hidden_states(sampled_hidden)
        layer_input = torch.cat([inputs_embeds, sampled_hidden], dim=-1)

        # Step 5: Position IDs and rotary embeddings
        position_ids = orig_positions.unsqueeze(0)

        # Step 6: Create flex attention mask
        mask_mod = create_peagle_mask_mod(
            anchor_pos=anchor_pos,
            depth=depth,
            lengths=lengths,
            total_seq_len=seq_length,
        )
        block_mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=total_sampled,
            KV_LEN=total_sampled,
            device=device,
        )

        # Step 7: Run through draft model layers
        cos, sin = self.draft_model.rotary_emb(
            layer_input, seq_len=position_ids.max().item() + 1
        )
        cos = cos.squeeze(0).squeeze(0)
        sin = sin.squeeze(0).squeeze(0)
        cos = cos[position_ids]
        sin = sin[position_ids]
        position_embeddings = (cos, sin)

        h = layer_input
        for layer in self.draft_model.layers:
            h = layer(h, block_mask, position_embeddings)

        # Step 8: Compute logits
        logits = self.draft_model.compute_logits(h)

        # Step 9: Compute loss and metrics (target is already logits from target model)
        loss, metrics = compute_peagle_metrics(
            logits=logits,
            targets=target,
            loss_mask=loss_mask,
            anchor_pos=anchor_pos,
            depth=depth,
            num_depths=self.num_depths,
            t2d=self.draft_model.t2d,
        )

        return loss, metrics
