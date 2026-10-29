import copy
from typing import Optional

import torch
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3RotaryEmbedding,
)
from typing_extensions import Tuple, Unpack

from specforge.modeling.draft.config import dflash as _draft_config
from specforge.modeling.draft.config.dflash import (
    DFlashConfig,
    resolve_dflash_causal_block,
)
from specforge.modeling.draft.models.layers.attention.backends.flex_options import (
    flex_attention_backend,
)
from specforge.modeling.draft.models.layers.attention.masks import (
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
    resolve_dflash_is_causal,
)
from specforge.modeling.draft.models.layers.kernels import (
    DEFAULT_DFLASH_KERNELS,
    DFlashKernels,
)
from specforge.modeling.draft.registry import register_draft

# Preserve the existing helper import paths while configuration owns the rules.
build_target_layer_ids = _draft_config.build_target_layer_ids
resolve_dflash_attention_layout = _draft_config.resolve_dflash_attention_layout
validate_dflash_attention_config = _draft_config.validate_dflash_attention_config
validate_dflash_mla_config = _draft_config.validate_dflash_mla_config

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"
_ATTENTION_MASK_SEMANTICS_VERSION = 2
_LEGACY_CAUSAL_ARCHITECTURES = {"DSparkDraftModel", "DSpineDraftModel"}


def dflash_attention_resume_contract(draft_model) -> dict:
    """Version only settings whose interpretation differs from legacy checkpoints.

    Global bidirectional attention is unchanged. DSpark and DSpine already
    supported global causal attention before independent window directions.
    """

    is_causal = resolve_dflash_causal_block(draft_model.config)
    if is_causal is None:
        return {}

    sliding_window = getattr(draft_model, "sliding_window", None)
    if sliding_window is None:
        if not is_causal:
            return {}
        architectures = getattr(draft_model.config, "architectures", None) or []
        if _LEGACY_CAUSAL_ARCHITECTURES.intersection(architectures):
            return {}

    return {
        "draft_attention_mask_semantics": (
            _ATTENTION_MASK_SEMANTICS_VERSION,
            tuple(draft_model.layer_types),
            sliding_window,
            is_causal,
        )
    }


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def _rope_config(config: Qwen3Config, attention_mode: str) -> Qwen3Config:
    """Rotary config for the mode: MLA rotates only the partial-RoPE slice."""

    if attention_mode != "mla":
        return config
    rope_config = copy.deepcopy(config)
    rope_config.head_dim = config.qk_rope_head_dim
    return rope_config


from .layers.attention import build_dflash_attention
from .layers.attention.dflash import (
    Qwen3DFlashAttentionBase,
    _prepare_dflash_eager_mask,
    apply_mla_rope,
    apply_rotary_pos_emb,
)
from .layers.attention.gqa import Qwen3DFlashAttention
from .layers.attention.mla import Qwen3DFlashMLAAttention
from .layers.ffn import build_ffn
from .layers.residual import build_residual


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        kernels: DFlashKernels,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = build_dflash_attention(
            config=config,
            layer_idx=layer_idx,
            kernels=kernels,
        )
        self.mlp = build_ffn(config, dense_factory=kernels.make_mlp)
        self.residual = build_residual(config)
        self.input_layernorm = kernels.make_rms_norm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = kernels.make_rms_norm(
            config.hidden_size, config.rms_norm_eps
        )

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
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(self.residual.branch_input(hidden_states))
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
        hidden_states = self.residual.combine(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(
            self.residual.branch_input(hidden_states)
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.residual.combine(residual, hidden_states)
        return hidden_states


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden


def normalize_draft_head_checkpoint_keys(
    module,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    """Map checkpoint-only nested head names onto the direct module layout.

    Early Domino/DSpark checkpoints saved their auxiliary heads beneath a
    ``logit_head`` container. The live architecture no longer owns that wrapper,
    but those tensors remain valid and must not be dropped during warm start or
    full resume.
    """

    del module, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    checkpoint_prefixes = (
        ("logit_head.prefix_gru.", "prefix_gru."),
        ("logit_head.embed_proj.", "embed_proj."),
        ("logit_head.markov_head.", "markov_head."),
        ("logit_head.confidence_head.", "confidence_head."),
    )
    for key in list(state_dict):
        if not key.startswith(prefix):
            continue
        local_key = key[len(prefix) :]
        for checkpoint_prefix, model_prefix in checkpoint_prefixes:
            if not local_key.startswith(checkpoint_prefix):
                continue
            normalized_key = prefix + model_prefix + local_key[len(checkpoint_prefix) :]
            if normalized_key not in state_dict:
                state_dict[normalized_key] = state_dict[key]
            state_dict.pop(key)
            break


@register_draft
class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    draft_options_class = DFlashConfig
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]
    decoder_layer_class = Qwen3DFlashDecoderLayer

    def __init__(
        self,
        config,
        dflash_kernels: Optional[DFlashKernels] = None,
    ) -> None:
        options = self.draft_options_class.from_model_config(config)
        config = options.to_model_config(config)
        super().__init__(config)
        self.draft_options = options
        self.layer_types = options.layer_types
        self.sliding_window = options.sliding_window
        self.attention_mode = options.attention_mode
        self.block_size = options.block_size
        kernels = dflash_kernels or DEFAULT_DFLASH_KERNELS
        dflash_config = config.dflash_config
        self.layers = nn.ModuleList(
            [
                self._build_decoder_layer(config, layer_idx, kernels)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.target_layer_ids = list(options.target_layer_ids)
        self.norm = kernels.make_rms_norm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(
            _rope_config(config, self.attention_mode)
        )
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = kernels.make_rms_norm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mask_token_id = options.mask_token_id
        self.projector_type = options.projector_type
        self.pure_draft_prefix_len = options.pure_draft_prefix_len
        self.shift_label = options.shift_label
        self._init_draft_head(config, dflash_config)
        self.register_load_state_dict_pre_hook(normalize_draft_head_checkpoint_keys)
        self.post_init()

    @classmethod
    def prepare_config(cls, config: Qwen3Config) -> Qwen3Config:
        """Validate the final architecture without allocating model parameters."""
        return cls.draft_options_class.from_model_config(config).to_model_config(config)

    def _build_decoder_layer(
        self,
        config: Qwen3Config,
        layer_idx: int,
        kernels: DFlashKernels,
    ) -> nn.Module:
        """Build one backbone layer; architecture variants override this seam."""

        return self.decoder_layer_class(config, layer_idx, kernels)

    def _init_draft_head(self, config, dflash_config: dict) -> None:
        del config, dflash_config

    def apply_logits_head(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
        prev_token_embeddings: Optional[torch.Tensor] = None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        del prev_token_ids, prev_token_embeddings, hidden_states
        return base_logits

    def apply_markov_logits(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.apply_logits_head(
            base_logits,
            prev_token_ids=prev_token_ids,
            hidden_states=hidden_states,
        )

    def predict_confidence(
        self,
        hidden_states: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        del hidden_states, prev_token_ids
        return None

    def _sample_draft_tokens(
        self,
        target: nn.Module,
        draft_hidden: torch.Tensor,
        block_output_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """Sample one speculative block from the draft-model hidden states.

        DFlash predicts the whole suffix in one LM-head call. Draft families
        with an auxiliary logits head can override this boundary without
        duplicating the target-cache and acceptance logic in ``spec_generate``.
        """
        del block_output_ids
        draft_logits = target.lm_head(draft_hidden[:, -self.block_size + 1 :, :])
        return sample(draft_logits)

    def _prepare_attention_mask(
        self,
        attention_mask,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        past_key_values: Optional[Cache],
        is_causal: Optional[bool] = None,
    ):
        """Return per-layer-type masks for one draft block when none is supplied.

        Training always passes masks. A mask-free call is one draft block after
        the cached plus new context (``spec_generate``); build masks matching
        the training masks. FA2 applies causality and the symmetric window
        natively, including when the query block is longer than the window.
        """
        if attention_mask is not None:
            return attention_mask
        implementation = self.config._attn_implementation
        if implementation not in ("eager", "sdpa", "flex_attention"):
            return None
        if is_causal is None:
            is_causal = getattr(self.config, "is_causal", None)

        # Count cached keys before any layer appends this call's context/draft.
        cached_length = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        context_length = cached_length + target_hidden.shape[1]
        batch_size, block_size = hidden_states.shape[:2]
        anchor_positions = torch.full(
            (batch_size, 1),
            context_length,
            dtype=torch.long,
            device=hidden_states.device,
        )
        mask_args = {
            "anchor_positions": anchor_positions,
            "block_keep_mask": torch.ones_like(anchor_positions, dtype=torch.bool),
            "S": context_length,
            "block_size": block_size,
            "device": hidden_states.device,
        }
        if implementation == "flex_attention":
            mask_builder = create_dflash_block_mask
            if flex_attention_backend() == "FLASH":
                # FLASH requires a minimum of this block size.
                mask_args["BLOCK_SIZE"] = (256, 128)
        else:
            mask_builder = create_dflash_sdpa_mask

        masks = {}
        if FULL_ATTENTION in self.layer_types:
            # A bidirectional full layer sees every key, so it needs no mask.
            full_causal = resolve_dflash_is_causal(is_causal, FULL_ATTENTION)
            masks[FULL_ATTENTION] = (
                mask_builder(**mask_args, is_causal=True) if full_causal else None
            )
        if SLIDING_ATTENTION in self.layer_types:
            masks[SLIDING_ATTENTION] = mask_builder(
                **mask_args,
                sliding_window=self.sliding_window,
                is_causal=is_causal,
            )
        return masks

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[object] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        attention_mask = self._prepare_attention_mask(
            attention_mask,
            hidden_states,
            target_hidden,
            past_key_values,
            is_causal=kwargs.get("is_causal"),
        )
        for layer_type, layer in zip(self.layer_types, self.layers):
            layer_attention_mask = (
                attention_mask[layer_type]
                if isinstance(attention_mask, dict)
                else attention_mask
            )
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=layer_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
    ):
        self.eval()
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens

        block_size = self.block_size
        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = torch.arange(
            output_ids.shape[1], device=target.device
        ).unsqueeze(0)

        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        # Prefill stage
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            output.logits, temperature
        )
        target_hidden = extract_context_feature(
            output.hidden_states, self.target_layer_ids
        )

        # Decode stage
        acceptance_lengths = []
        start = input_ids.shape[1]
        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[:, start : start + block_size]
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_hidden = self(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[
                    :, past_key_values_draft.get_seq_length() : start + block_size
                ],
                past_key_values=past_key_values_draft,
                use_cache=True,
            )
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = self._sample_draft_tokens(
                target,
                draft_hidden,
                block_output_ids,
            )

            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )

            posterior = sample(output.logits, temperature)
            acceptance_length = (
                (block_output_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = posterior[
                :, acceptance_length
            ]
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            target_hidden = extract_context_feature(
                output.hidden_states, self.target_layer_ids
            )[:, : acceptance_length + 1, :]
            acceptance_lengths.append(acceptance_length + 1)
            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:]
                for stop_token_id in stop_token_ids
            ):
                break
        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids is not None:
            stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(
                output_ids[0][num_input_tokens:], stop_token_ids
            ).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[
                    :, : num_input_tokens + stop_token_indices[0] + 1
                ]

        return output_ids


# Preserve the historical helper surface while implementations live in layers.
__all__ = [
    "DFlashDraftModel",
    "Qwen3DFlashDecoderLayer",
    "Qwen3DFlashAttentionBase",
    "Qwen3DFlashAttention",
    "Qwen3DFlashMLAAttention",
    "apply_rotary_pos_emb",
    "apply_mla_rope",
    "_prepare_dflash_eager_mask",
    "build_target_layer_ids",
    "extract_context_feature",
    "sample",
]
