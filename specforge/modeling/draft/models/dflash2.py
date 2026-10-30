"""DFlash 2 draft architecture.

DFlash 2 keeps the one-pass DFlash backbone and adds two checkpoint-driven
components used by the serving implementation:

* a grouped, dynamic depthwise convolution around every attention and MLP;
* a low-rank candidate selector that re-ranks the target head's top-k tokens.

The module and parameter names intentionally match SGLang's
``DFlash2DraftModel`` so a normal Hugging Face export can be served directly.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import FlashAttentionKwargs, Qwen3Config
from typing_extensions import Tuple, Unpack

from specforge.config.draft.dflash2 import DFlash2Config
from specforge.modeling.draft.models.layers.kernels import DFlashKernels
from specforge.modeling.draft.registry import register_draft

from .dflash import DFlashDraftModel, Qwen3DFlashDecoderLayer
from .layers.convolution.dflash2 import DFlashGroupedConv


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

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(self.residual.branch_input(hidden_states))
        hidden_states, attention_kernel = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = self.attention_conv.finish(
            hidden_states,
            attention_kernel,
        )
        hidden_states = self.residual.combine(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(
            self.residual.branch_input(hidden_states)
        )
        hidden_states, mlp_kernel = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, mlp_kernel)
        return self.residual.combine(residual, hidden_states)


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


@register_draft
class DFlash2DraftModel(DFlashDraftModel):
    """DFlash backbone with local convolution and candidate path selection."""

    _no_split_modules = ["Qwen3DFlash2DecoderLayer"]
    decoder_layer_class = Qwen3DFlash2DecoderLayer

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        if isinstance(module, DFlashGroupedConv):
            # DFlashDraftModel.post_init() reinitializes child Linear modules.
            # Restore the additive dynamic-kernel branch to an exact no-op after
            # that pass; checkpoint loading can still overwrite these weights.
            nn.init.zeros_(module.kernel_projection.weight)

    draft_options_class = DFlash2Config

    def _build_decoder_layer(
        self,
        config: Qwen3Config,
        layer_idx: int,
        kernels: DFlashKernels,
    ) -> nn.Module:
        from specforge.config.draft.component_registry import build_component

        def grouped_conv():
            return build_component(
                "conv", config.layers[layer_idx].conv, "draft_block", config=config
            )

        return self.decoder_layer_class(
            config,
            layer_idx,
            kernels,
            attention_conv=grouped_conv(),
            mlp_conv=grouped_conv(),
        )

    def _init_draft_head(self, config: Qwen3Config, dflash_config: dict) -> None:
        selector_rank = dflash_config["selector_rank"]
        selector_top_k = dflash_config["selector_top_k"]
        self.candidate_selector = CandidateSelector(
            hidden_size=int(config.hidden_size),
            vocab_size=int(config.vocab_size),
            state_rank=selector_rank,
            top_k=selector_top_k,
            initializer_range=float(config.initializer_range),
        )

    def unary_logits_transform_is_identity(self) -> bool:
        """Whether ``transform_unary_logits`` only upcasts to FP32."""

        return (
            self.config.dflash_config["output_multiplier"] == 1.0
            and self.config.dflash_config["final_logit_softcapping"] is None
        )

    def transform_unary_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the public DFlash2 unary-logit transform used by SGLang."""

        if self.unary_logits_transform_is_identity():
            # ``x * 1.0 == x`` bit for bit; skip the extra FP32 logit tensor.
            return logits.float()
        transformed = logits.float() * self.config.dflash_config["output_multiplier"]
        softcap = self.config.dflash_config["final_logit_softcapping"]
        if softcap is not None:
            transformed = torch.tanh(transformed / softcap) * softcap
        return transformed

    def _sample_draft_tokens(
        self,
        target: nn.Module,
        draft_hidden: torch.Tensor,
        block_output_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        hidden = draft_hidden[:, -self.config.block_size + 1 :, :]
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
