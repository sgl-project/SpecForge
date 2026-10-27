"""LongSpark draft model.

The drafter reads captured target K/V and hidden states and predicts a block of
tokens in parallel, without keeping a draft KV cache. The target embedding,
final norm and LM head are owned by the training wrapper and are not part of
this model's state dict.
"""

import re

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig, PreTrainedModel
from transformers import initialization as init
from transformers.conversion_mapping import register_checkpoint_conversion_mapping
from transformers.core_model_loading import WeightRenaming

from .dspark import VanillaMarkovHead
from .longspark_context import GlobalMemory, TargetContext
from .longspark_layers import (
    DraftBlockInput,
    DraftStateEncoder,
    LongSparkDecoderLayer,
    RMSNorm,
)
from .registry import register_draft

# Parameter-name prefixes used by checkpoints of the reference implementation.
_LEGACY_KEY_PREFIXES = (
    (
        "global16_raw256_reference.global_query_generator.learned_query",
        "global_memory.queries",
    ),
    ("draft_state_encoder.", "state_encoder."),
    ("parallel_shifted_input.", "block_input."),
    ("parallel_qwen_layers.", "layers."),
    ("parallel_markov_head.", "markov_head."),
)

# Rename legacy keys in ``from_pretrained``; ``load_state_dict`` uses the hook below.
register_checkpoint_conversion_mapping(
    "longspark",
    [
        WeightRenaming(f"^{re.escape(legacy)}", current)
        for legacy, current in _LEGACY_KEY_PREFIXES
    ],
    overwrite=True,
)


def _rename_legacy_keys(
    module,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    """Load checkpoints saved with the reference implementation's names."""
    del module, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    for key in list(state_dict):
        if not key.startswith(prefix):
            continue
        local_key = key[len(prefix) :]
        for legacy, current in _LEGACY_KEY_PREFIXES:
            if local_key.startswith(legacy):
                new_key = prefix + current + local_key[len(legacy) :]
                state_dict.setdefault(new_key, state_dict.pop(key))
                break


class LongSparkConfig(PretrainedConfig):
    model_type = "longspark"

    def __init__(
        self,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=5,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=151936,
        target_layer_ids=None,
        block_size=7,
        mask_token_id=151669,
        global_memory_slots=16,
        recent_kv_window=256,
        markov_rank=256,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        gradient_checkpointing=True,
        initializer_range=0.02,
        **kwargs,
    ):
        # Stored as ``qwen_gradient_checkpointing``: PreTrainedModel consumes and
        # deletes ``config.gradient_checkpointing``. Both names are accepted.
        gradient_checkpointing = kwargs.pop(
            "qwen_gradient_checkpointing", gradient_checkpointing
        )
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.target_layer_ids = list(
            [0, 9, 18, 26, 35] if target_layer_ids is None else target_layer_ids
        )
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.global_memory_slots = global_memory_slots
        self.recent_kv_window = recent_kv_window
        self.markov_rank = markov_rank
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.qwen_gradient_checkpointing = gradient_checkpointing
        self.initializer_range = initializer_range
        if num_hidden_layers != len(self.target_layer_ids):
            raise ValueError("one draft layer is required per selected target layer")
        if len(set(self.target_layer_ids)) != len(self.target_layer_ids):
            raise ValueError("target layer ids must be unique")
        for name in ("block_size", "global_memory_slots", "recent_kv_window"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= mask_token_id < vocab_size:
            raise ValueError("mask token must be in the target vocabulary")
        if markov_rank <= 0:
            raise ValueError("markov_rank must be positive")


@register_draft
class LongSparkDraftModel(PreTrainedModel):
    config_class = LongSparkConfig
    _no_split_modules = ["LongSparkDecoderLayer"]
    _supports_sdpa = True

    def __init__(self, config: LongSparkConfig):
        super().__init__(config)
        self.target_layer_ids = list(config.target_layer_ids)
        self.block_size = config.block_size
        self.mask_token_id = config.mask_token_id
        self.state_encoder = DraftStateEncoder(
            config.num_hidden_layers, config.hidden_size
        )
        self.block_input = DraftBlockInput(
            config.hidden_size, config.block_size, config.mask_token_id
        )
        self.layers = nn.ModuleList(
            LongSparkDecoderLayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                rope_theta=config.rope_theta,
                rms_norm_eps=config.rms_norm_eps,
            )
            for _ in self.target_layer_ids
        )
        self.global_memory = GlobalMemory(
            num_layers=len(self.target_layer_ids),
            num_heads=config.num_attention_heads,
            num_slots=config.global_memory_slots,
            head_dim=config.head_dim,
        )
        self.markov_head = VanillaMarkovHead(
            vocab_size=config.vocab_size, markov_rank=config.markov_rank
        )
        self.register_load_state_dict_pre_hook(_rename_legacy_keys)
        self.post_init()

    def _init_weights(self, module):
        # The ``init`` helpers skip tensors already loaded by from_pretrained.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            init.ones_(module.weight)

    def forward(
        self,
        *,
        keys: torch.Tensor,
        values: torch.Tensor,
        source_indices: torch.Tensor,
        anchor_positions: torch.Tensor,
        current_hidden_states: torch.Tensor,
        current_token_ids: torch.Tensor,
        target_embed_tokens: nn.Module,
    ) -> torch.Tensor:
        """Return draft hidden states ``[anchors, block_size, hidden]``.

        Anchor ``a`` sees target tokens ``[0, a)``, the selected target hidden
        states at ``a - 1`` and the token ``x[a]``; its labels are
        ``x[a + 1 : a + 1 + block_size]``.
        """
        anchors = source_indices.numel()
        expected_hidden = (anchors, len(self.target_layer_ids), self.config.hidden_size)
        if current_hidden_states.shape != expected_hidden:
            raise ValueError(f"current hidden states must have shape {expected_hidden}")
        if current_token_ids.shape != (anchors,) or anchor_positions.shape != (
            anchors,
        ):
            raise ValueError("current tokens and positions must match anchor count")
        if keys.dtype != next(self.parameters()).dtype or values.dtype != keys.dtype:
            raise ValueError("target K/V must already match the drafter dtype")
        if (
            keys.requires_grad
            or values.requires_grad
            or current_hidden_states.requires_grad
        ):
            raise ValueError("target features must be frozen")
        if torch.any(anchor_positions <= 0) or torch.any(
            anchor_positions >= keys.shape[-2]
        ):
            raise ValueError("anchors must be strictly inside their target sequence")
        prefix_mask = (
            torch.arange(keys.shape[-2], device=keys.device)[None, :]
            < anchor_positions[:, None]
        )
        context = TargetContext(
            global_memory=self.global_memory,
            keys=keys,
            values=values,
            prefix_mask=prefix_mask,
            source_indices=source_indices,
            window=self.config.recent_kv_window,
        )
        latest = self.state_encoder(current_hidden_states)
        states = self.block_input(
            current_token_ids=current_token_ids,
            latest_target_state=latest,
            target_embed_tokens=target_embed_tokens,
            model_dtype=keys.dtype,
        )
        positions = (
            anchor_positions[:, None]
            + torch.arange(self.block_size, device=keys.device)[None, :]
        )
        for index, layer in enumerate(self.layers):

            def apply_layer(hidden, decoder=layer, layer_index=index):
                return decoder(
                    hidden,
                    position_ids=positions,
                    context=context,
                    layer_index=layer_index,
                )

            if (
                self.config.qwen_gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                states = checkpoint(apply_layer, states, use_reentrant=False)
            else:
                states = apply_layer(states)
        return states

    def project_logits(
        self,
        states: torch.Tensor,
        *,
        target_final_norm: nn.Module,
        target_lm_head: nn.Module,
        previous_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the frozen target norm and LM head, plus the Markov bias."""
        norm_input = states.reshape(-1, states.shape[-1]).to(
            target_final_norm.weight.dtype
        )
        norm_output = F.rms_norm(
            norm_input,
            (states.shape[-1],),
            target_final_norm.weight.detach(),
            float(target_final_norm.variance_epsilon),
        )
        normalized = norm_output.reshape_as(states).to(states.dtype)
        logits = F.linear(
            normalized.to(target_lm_head.weight.dtype),
            target_lm_head.weight.detach(),
        ).float()
        return self.markov_head.apply_block_logits(
            logits, token_ids=previous_token_ids, hidden_states=None
        )
