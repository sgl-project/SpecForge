"""DSpine adjacent injection and predecessor-conditioned candidate decoding."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from specforge.config.draft.dspine import DSpineDraftConfig
from specforge.modeling.draft.registry import register_draft

from .dflash import DFlashDraftModel


class DSpineOutput(NamedTuple):
    hidden_states: torch.Tensor
    pre_injection: torch.Tensor
    layer_features: torch.Tensor


class AdjacentInjection(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, config: Mapping):
        super().__init__()
        self.rank = config["transfer_rank"]
        self.config = config
        self.readouts = nn.ModuleList(
            nn.Linear(hidden_size, self.rank, bias=False) for _ in range(num_layers)
        )
        self.writes = nn.ModuleList(
            nn.Linear(config["message_size"], hidden_size, bias=False)
            for _ in range(num_layers)
        )
        self.message = nn.Linear(self.rank, config["message_size"], bias=False)
        self.receiver_gate = nn.Linear(self.rank, config["message_size"], bias=False)
        self.predecessor_gate = nn.Linear(self.rank, config["message_size"], bias=False)
        self.bias = nn.Parameter(torch.zeros(config["message_size"]))
        for write in self.writes:
            nn.init.zeros_(write.weight)
        nn.init.zeros_(self.predecessor_gate.weight)

    def feature(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        projected = self.readouts[layer](hidden)
        return (math.sqrt(self.rank) * F.normalize(projected.float(), dim=-1)).to(
            hidden.dtype
        )

    def write(
        self,
        hidden: torch.Tensor,
        receiver: torch.Tensor,
        predecessor: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        # FSDP may keep the token-code buffer in FP32 while projections use BF16.
        predecessor = predecessor.to(receiver.dtype)
        gate = torch.sigmoid(
            self.receiver_gate(receiver)
            + self.predecessor_gate(predecessor)
            + self.bias
        )
        delta = self.writes[layer](gate * self.message(predecessor))
        scale = (
            torch.linalg.vector_norm(hidden.float(), dim=-1, keepdim=True)
            / math.sqrt(hidden.shape[-1])
        ).to(hidden.dtype)
        return hidden + scale * delta

    def forward(
        self,
        hidden: torch.Tensor,
        layer: int,
        token_ids: torch.Tensor,
        codes: torch.Tensor,
        replace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.feature(hidden, layer)
        anchor = math.sqrt(self.rank) * F.embedding(token_ids[..., :1], codes)
        predecessors = torch.cat((anchor, features[..., 1:-1, :]), dim=-2)
        if replace is not None and layer < self.config["replacement_layers"]:
            correct = math.sqrt(self.rank) * F.embedding(token_ids[..., :-1], codes)
            predecessors = torch.where(replace[..., None, None], correct, predecessors)
        successors = self.write(
            hidden[..., 1:, :], features[..., 1:, :], predecessors, layer
        )
        return torch.cat((hidden[..., :1, :], successors), dim=-2)


@register_draft
class DSpineDraftModel(DFlashDraftModel):
    draft_options_class = DSpineDraftConfig

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        config = self.config
        self.injection = AdjacentInjection(
            config.hidden_size, config.num_hidden_layers, self.config.dspine_config
        )
        self.register_buffer(
            "transfer_codes",
            torch.zeros(config.vocab_size, self.config.dspine_config["transfer_rank"]),
        )
        self.register_buffer("transfer_ready", torch.tensor(False))

    @torch.no_grad()
    def initialize_transfer_space(
        self, head_weight: torch.Tensor, chunk_size: int = 4096
    ) -> None:
        """Initialize fixed whitened token codes and the layer readouts from the target head."""
        if tuple(head_weight.shape) != (
            self.config.vocab_size,
            self.config.hidden_size,
        ):
            raise ValueError(
                "DSpine transfer space requires the target head's full vocabulary and hidden width"
            )
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        width = head_weight.shape[1]
        covariance = torch.zeros(
            width, width, device=head_weight.device, dtype=torch.float32
        )
        total = covariance.new_zeros(width)
        for rows in head_weight.split(chunk_size):
            normalized = F.normalize(rows.float(), dim=-1)
            total.add_(normalized.sum(dim=0))
            covariance.add_(normalized.T @ normalized)
        mean = total / head_weight.shape[0]
        covariance.sub_(head_weight.shape[0] * torch.outer(mean, mean))
        values, directions = torch.linalg.eigh(covariance)
        values = values[-self.config.dspine_config["transfer_rank"] :]
        directions = directions[:, -self.config.dspine_config["transfer_rank"] :]
        if not torch.isfinite(values).all() or (values <= 0).any():
            raise ValueError(
                "Target head has insufficient positive principal directions for transfer_rank"
            )
        for start in range(0, head_weight.shape[0], chunk_size):
            rows = head_weight[start : start + chunk_size]
            codes = F.normalize(
                (F.normalize(rows.float(), dim=-1) - mean) @ directions / values.sqrt(),
                dim=-1,
            )
            self.transfer_codes[start : start + len(rows)].copy_(codes)
        for readout in self.injection.readouts:
            readout.weight.copy_(directions.T)
        self.transfer_ready.fill_(True)

    def forward(
        self,
        position_ids: torch.Tensor,
        attention_mask,
        noise_embedding: torch.Tensor,
        target_hidden: torch.Tensor,
        block_token_ids: torch.Tensor,
        replace_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> DSpineOutput:
        torch._assert_async(
            self.transfer_ready,
            "Initialize or load the DSpine transfer space before use",
        )
        if (
            kwargs.pop("use_cache", False)
            or kwargs.pop("past_key_values", None) is not None
        ):
            raise ValueError(
                "DSpine reference forward requires explicit context without a draft KV cache"
            )
        if attention_mask is None:
            raise ValueError("DSpine requires an explicit causal block attention mask")
        batch, blocks, size = block_token_ids.shape
        if size != self.config.block_size or noise_embedding.shape[:2] != (
            batch,
            blocks * size,
        ):
            raise ValueError(
                "DSpine token blocks and noise embeddings must match block_size"
            )
        hidden = noise_embedding
        context = self.hidden_norm(self.fc(target_hidden))
        layer_features = []
        for index, (spec, layer) in enumerate(zip(self.config.layers, self.layers)):
            layer_type = spec.attention.layer_type
            rotary = self.rotary_emb if index == 0 else self.extra_rotary_emb[index - 1]
            positions = rotary(hidden, position_ids)
            mask = (
                (
                    attention_mask[index]
                    if index in attention_mask
                    else attention_mask[layer_type]
                )
                if isinstance(attention_mask, dict)
                else attention_mask
            )
            hidden = layer(
                hidden_states=hidden,
                target_hidden=context,
                attention_mask=mask,
                position_ids=position_ids,
                position_embeddings=positions,
                **kwargs,
            )
            pre_injection = hidden.reshape(batch, blocks, size, -1)
            hidden = self.injection(
                pre_injection, index, block_token_ids, self.transfer_codes, replace_mask
            )
            layer_features.append(self.injection.feature(hidden, index))
            hidden = hidden.reshape(batch, blocks * size, -1)
        return DSpineOutput(
            self.norm(hidden).reshape(batch, blocks, size, -1),
            pre_injection,
            torch.stack(layer_features),
        )

    def refine(
        self, hidden: torch.Tensor, predecessor_ids: torch.Tensor
    ) -> torch.Tensor:
        torch._assert_async(
            self.transfer_ready,
            "Initialize or load the DSpine transfer space before use",
        )
        layer = len(self.layers) - 1
        receiver = self.injection.feature(hidden, layer)
        predecessor = math.sqrt(self.injection.rank) * F.embedding(
            predecessor_ids, self.transfer_codes
        )
        return self.norm(self.injection.write(hidden, receiver, predecessor, layer))

    def transition_scores(
        self,
        pre_injection: torch.Tensor,
        candidate_ids: torch.Tensor,
        anchor_ids: torch.Tensor,
        head_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Score all adjacent candidate pairs from cached pre-injection states."""
        count = candidate_ids.shape[-1]
        anchor = anchor_ids[..., None, None].expand(*anchor_ids.shape, 1, count)
        predecessors = torch.cat((anchor, candidate_ids[..., :-1, :]), dim=-2)
        hidden = pre_injection.unsqueeze(-2).expand(
            *pre_injection.shape[:-1], count, pre_injection.shape[-1]
        )
        refined = self.refine(hidden, predecessors)
        weights = F.embedding(candidate_ids, head_weight)
        return torch.einsum("...ikh,...ijh->...ikj", refined, weights).float()

    @staticmethod
    def select_cached(
        scores: torch.Tensor, candidate_ids: torch.Tensor
    ) -> torch.Tensor:
        index = torch.zeros(
            candidate_ids.shape[:-2], device=candidate_ids.device, dtype=torch.long
        )
        selected = []
        for position in range(candidate_ids.shape[-2]):
            row = (
                scores[..., position, :, :]
                .gather(
                    -2, index[..., None, None].expand(*index.shape, 1, scores.shape[-1])
                )
                .squeeze(-2)
            )
            index = row.argmax(dim=-1)
            selected.append(
                candidate_ids[..., position, :].gather(-1, index[..., None]).squeeze(-1)
            )
        return torch.stack(selected, dim=-1)

    def spec_generate(self, *args, **kwargs):
        raise NotImplementedError(
            "DSpine supports draft training and transition scoring; native serving requires a DSpine verifier integration"
        )
