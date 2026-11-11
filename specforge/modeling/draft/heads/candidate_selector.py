"""Low-rank candidate scoring, independent of the draft backbone."""

from __future__ import annotations

import torch
from torch import nn


class CandidateSelector(nn.Module):
    """Low-rank transition scorer used to select a coherent token path."""

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        state_rank: int,
        top_k: int,
        initializer_range: float,
    ) -> None:
        super().__init__()
        if state_rank < 1:
            raise ValueError("DFlash2 selector_rank must be >= 1")
        if top_k < 1 or top_k > vocab_size:
            raise ValueError(
                "DFlash2 selector_top_k must be in [1, vocab_size], got "
                f"{top_k} for vocab_size={vocab_size}"
            )
        self.top_k = int(top_k)
        self.predecessor_codebook = nn.Parameter(
            torch.empty(int(vocab_size), int(state_rank))
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(int(vocab_size), int(state_rank))
        )
        self.hidden_projection = nn.Linear(
            int(hidden_size),
            int(state_rank),
            bias=False,
        )
        nn.init.normal_(self.predecessor_codebook, std=initializer_range)
        # The transition must start as a no-op so a fresh DFlash2 selector is
        # numerically identical to the unary DFlash proposal. The successor
        # factor learns first; the other bilinear factors receive signal once it
        # becomes non-zero.
        nn.init.zeros_(self.successor_codebook)

    def score_candidates(
        self,
        *,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Add low-rank predecessor transitions to a candidate set's logits."""

        predecessor = self.predecessor_codebook[predecessor_ids]
        successor = self.successor_codebook[candidate_ids]
        context = predecessor * self.hidden_projection(hidden_states)
        transition = torch.einsum("...r,...kr->...k", context, successor)
        return unary_logits + transition

    def build_lattice(
        self,
        *,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Build SGLang's K-by-K transition lattice for every proposal slot."""

        predecessor_ids = torch.cat(
            [
                anchor_token_ids[:, None, None].expand(-1, 1, self.top_k),
                candidate_ids[:, :-1],
            ],
            dim=1,
        )
        predecessor = self.predecessor_codebook[predecessor_ids]
        successor = self.successor_codebook[candidate_ids]
        context = predecessor * self.hidden_projection(hidden_states)[:, :, None]
        return unary_logits[:, :, None] + torch.einsum(
            "blpr,blcr->blpc",
            context,
            successor,
        )

    def greedy_path(
        self,
        *,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Walk the candidate lattice for the local greedy inference helper."""

        predecessor_ids = anchor_token_ids
        path = []
        for position in range(candidate_ids.shape[1]):
            scores = self.score_candidates(
                candidate_ids=candidate_ids[:, position],
                unary_logits=unary_logits[:, position],
                hidden_states=hidden_states[:, position],
                predecessor_ids=predecessor_ids,
            )
            selected = scores.argmax(dim=-1, keepdim=True)
            predecessor_ids = candidate_ids[:, position].gather(1, selected)[:, 0]
            path.append(predecessor_ids)
        return torch.stack(path, dim=1)
