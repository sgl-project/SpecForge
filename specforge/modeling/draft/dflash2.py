"""DFlash 2 draft architecture.

DFlash 2 keeps the one-pass DFlash backbone and adds two checkpoint-driven
components used by the serving implementation:

* a grouped, dynamic depthwise convolution around every attention and MLP;
* a low-rank candidate selector that re-ranks the target head's top-k tokens.

The module and parameter names intentionally match SGLang's
``DFlash2DraftModel`` so a normal Hugging Face export can be served directly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from .dflash import DFlashDraftModel, Qwen3DFlashDecoderLayer
from .dflash_kernels import DFlashKernels
from .registry import register_draft


class DFlashGroupedConv(nn.Module):
    """Grouped dynamic depthwise convolution within each proposal block.

    One projection of the sublayer input produces the dynamic kernel deltas
    for both the input and output side.  ``base_kernel`` starts as an identity,
    which makes enabling the module a stable extension of a DFlash backbone.
    """

    def __init__(
        self,
        hidden_size: int,
        block_size: int,
        taps: int,
        group_size: int,
    ) -> None:
        super().__init__()
        if taps < 1:
            raise ValueError("DFlash2 conv_kernel_size must be >= 1")
        if taps > block_size:
            raise ValueError(
                "DFlash2 conv_kernel_size must not exceed block_size, got "
                f"conv_kernel_size={taps}, block_size={block_size}"
            )
        if group_size < 1 or hidden_size % group_size:
            raise ValueError(
                f"DFlash2 conv_group_size={group_size} must divide "
                f"hidden_size={hidden_size}"
            )

        self.block_size = int(block_size)
        self.taps = int(taps)
        self.group_size = int(group_size)
        self.num_groups = int(hidden_size) // self.group_size

        # [input/output side, tap, channel], matching SGLang's loader contract.
        base_kernel = torch.zeros(2, self.taps, int(hidden_size))
        base_kernel[:, 0] = 1.0
        self.base_kernel = nn.Parameter(base_kernel)
        self.kernel_projection = nn.Linear(
            int(hidden_size),
            2 * self.taps * self.num_groups,
            bias=False,
        )

    def _convolve(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        *,
        side: int,
    ) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        if sequence_length % self.block_size:
            raise ValueError(
                "DFlash2 convolution sequence length must be divisible by "
                f"block_size={self.block_size}, got {sequence_length}"
            )

        num_blocks = sequence_length // self.block_size
        blocks = hidden_states.reshape(
            batch_size,
            num_blocks,
            self.block_size,
            self.num_groups,
            self.group_size,
        )
        dynamic = delta.reshape(
            batch_size,
            num_blocks,
            self.block_size,
            self.taps,
            self.num_groups,
        )
        base = self.base_kernel[side].reshape(
            1,
            1,
            1,
            self.taps,
            self.num_groups,
            self.group_size,
        )
        coefficients = base + dynamic.unsqueeze(-1)

        output = coefficients[:, :, :, 0] * blocks
        for tap in range(1, self.taps):
            shifted = F.pad(
                blocks[:, :, : self.block_size - tap],
                (0, 0, 0, 0, tap, 0),
            )
            output = output + coefficients[:, :, :, tap] * shifted
        return output.reshape(batch_size, sequence_length, hidden_size)

    def prepare(self, hidden_states: torch.Tensor):
        coefficients = self.kernel_projection(hidden_states).reshape(
            *hidden_states.shape[:-1],
            2,
            self.taps,
            self.num_groups,
        )
        return (
            self._convolve(hidden_states, coefficients[..., 0, :, :], side=0),
            coefficients[..., 1, :, :],
        )

    def finish(
        self,
        hidden_states: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, side=1)


class Qwen3DFlash2DecoderLayer(Qwen3DFlashDecoderLayer):
    """Qwen3 DFlash layer with DFlash2 convolutional sublayer wrappers."""

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        kernels: DFlashKernels,
        *,
        attention_conv: DFlashGroupedConv,
        mlp_conv: DFlashGroupedConv,
    ) -> None:
        super().__init__(config, layer_idx, kernels)
        self.attention_conv = attention_conv
        self.mlp_conv = mlp_conv

    def forward(self, **kwargs):
        target_hidden = kwargs.get("target_hidden")
        hidden_states = kwargs.get("hidden_states")
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attention_kernel = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=kwargs.get("attention_mask"),
            position_ids=kwargs.get("position_ids"),
            past_key_values=kwargs.get("past_key_value"),
            output_attentions=kwargs.get("output_attentions", False),
            use_cache=kwargs.get("use_cache", False),
            cache_position=kwargs.get("cache_position"),
            position_embeddings=kwargs.get("position_embeddings"),
            **{
                key: value
                for key, value in kwargs.items()
                if key
                not in {
                    "target_hidden",
                    "hidden_states",
                    "attention_mask",
                    "position_ids",
                    "past_key_value",
                    "output_attentions",
                    "use_cache",
                    "cache_position",
                    "position_embeddings",
                }
            },
        )[0]
        hidden_states = self.attention_conv.finish(
            hidden_states,
            attention_kernel,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, mlp_kernel = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, mlp_kernel)
        return residual + hidden_states


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
        nn.init.normal_(self.successor_codebook, std=initializer_range)

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


@register_draft
class DFlash2DraftModel(DFlashDraftModel):
    """DFlash backbone with local convolution and candidate path selection."""

    _no_split_modules = ["Qwen3DFlash2DecoderLayer"]
    decoder_layer_class = Qwen3DFlash2DecoderLayer

    def _dflash2_config(self) -> dict:
        return dict(getattr(self.config, "dflash_config", None) or {})

    def _build_decoder_layer(
        self,
        config: Qwen3Config,
        layer_idx: int,
        kernels: DFlashKernels,
    ) -> nn.Module:
        method_config = dict(getattr(config, "dflash_config", None) or {})
        taps = method_config.get("conv_kernel_size")
        group_size = method_config.get("conv_group_size")
        if not isinstance(taps, int) or isinstance(taps, bool):
            raise ValueError(
                "DFlash2DraftModel requires dflash_config.conv_kernel_size"
            )
        if not isinstance(group_size, int) or isinstance(group_size, bool):
            raise ValueError("DFlash2DraftModel requires dflash_config.conv_group_size")

        def grouped_conv() -> DFlashGroupedConv:
            return DFlashGroupedConv(
                hidden_size=int(config.hidden_size),
                block_size=self.block_size,
                taps=taps,
                group_size=group_size,
            )

        return self.decoder_layer_class(
            config,
            layer_idx,
            kernels,
            attention_conv=grouped_conv(),
            mlp_conv=grouped_conv(),
        )

    def _init_draft_head(self, config: Qwen3Config, dflash_config: dict) -> None:
        selector_rank = dflash_config.get("selector_rank")
        selector_top_k = dflash_config.get("selector_top_k")
        if not isinstance(selector_rank, int) or isinstance(selector_rank, bool):
            raise ValueError("DFlash2DraftModel requires dflash_config.selector_rank")
        if not isinstance(selector_top_k, int) or isinstance(selector_top_k, bool):
            raise ValueError("DFlash2DraftModel requires dflash_config.selector_top_k")
        self.candidate_selector = CandidateSelector(
            hidden_size=int(config.hidden_size),
            vocab_size=int(config.vocab_size),
            state_rank=selector_rank,
            top_k=selector_top_k,
            initializer_range=float(config.initializer_range),
        )

    def transform_unary_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the public DFlash2 unary-logit transform used by SGLang."""

        method_config = self._dflash2_config()
        transformed = logits.float() * float(
            method_config.get("output_multiplier", 1.0)
        )
        softcap = method_config.get("final_logit_softcapping")
        if softcap is not None:
            softcap = float(softcap)
            if softcap <= 0:
                raise ValueError("DFlash2 final_logit_softcapping must be > 0")
            transformed = torch.tanh(transformed / softcap) * softcap
        return transformed

    def _sample_draft_tokens(
        self,
        target: nn.Module,
        draft_hidden: torch.Tensor,
        block_output_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        hidden = draft_hidden[:, -self.block_size + 1 :, :]
        unary_logits = self.transform_unary_logits(target.lm_head(hidden))
        unary_topk, candidate_ids = unary_logits.topk(
            self.candidate_selector.top_k,
            dim=-1,
        )
        return self.candidate_selector.greedy_path(
            candidate_ids=candidate_ids,
            unary_logits=unary_topk,
            hidden_states=hidden,
            anchor_token_ids=block_output_ids[:, 0],
        )


__all__ = [
    "CandidateSelector",
    "DFlash2DraftModel",
    "DFlashGroupedConv",
    "Qwen3DFlash2DecoderLayer",
]
