"""Model-owned recurrent execution with caller-scoped, differentiable KV caches."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from specforge.modeling.draft.layers.attention.backends import (
    attention_backend,
    execute_attention,
)
from specforge.modeling.draft.layers.attention.masks import (
    AttentionLayout,
    prepare_attention_mask,
)
from specforge.modeling.draft.layers.rope import apply_rotary


class FullEagleRollout:
    """General recurrent execution for components without a projected KV cache."""

    def __init__(self, model):
        self.model = model
        self.tokens, self.features, self.positions = [], [], []

    def step(self, input_ids, features, padding, positions):
        from ..layers.inputs import LayerInputs

        depth, length = len(self.tokens), input_ids.shape[1]
        self.tokens.append(input_ids)
        self.features.append(features)
        self.positions.append(positions + depth)
        all_positions = torch.cat(self.positions, dim=-1)
        logical = all_positions[0] if all_positions.ndim == 3 else all_positions
        valid_length = padding.sum(-1)

        def visible(b, h, q, k):
            qa, ka = q % length, k % length
            qd, kd = q // length, k // length
            return (
                (((kd == 0) & (ka <= qa)) | ((ka == qa) & (kd <= qd)))
                & (qa + qd < valid_length[b])
                & (ka + kd < valid_length[b])
            )

        layout = AttentionLayout(
            query_positions=logical,
            key_positions=logical,
            visibility=visible,
            active_queries=(
                torch.arange(all_positions.shape[-1], device=padding.device)[None]
                % length
                + torch.arange(all_positions.shape[-1], device=padding.device)[None]
                // length
                < valid_length[:, None]
            ),
        )
        output = self.model(
            torch.cat(self.tokens, dim=1),
            None,
            (LayerInputs(position_ids=all_positions, attention_layout=layout),),
            recurrent_hidden=torch.cat(self.features, dim=1),
            output_logits=False,
        )
        return output.hidden_states[:, -length:]


class EagleRollout:
    def __init__(self, model, backend, *, ring_group=None, ulysses_group=None):
        self.model, self.backend = model, backend
        self.ring_group, self.ulysses_group = ring_group, ulysses_group
        self.keys, self.values = (), ()

    def step(self, input_ids, features, padding, positions):
        input_embeds = self.model.embed_tokens(input_ids)
        hidden_states, attention_mask, position_ids = features, padding, positions
        # Capture immutable cache tuples before checkpointing. Recompute must
        # neither append a branch twice nor observe a later rollout's cache.
        previous_keys, previous_values = self.keys, self.values

        cached_step = EagleCachedStep(
            backend=self.backend,
            padding=attention_mask,
            positions=position_ids,
            keys=previous_keys,
            values=previous_values,
            rotary=self.model.backbone.rotary[0],
            ring_group=self.ring_group,
            ulysses_group=self.ulysses_group,
        )

        def run(embeds, features):
            # Enter the stage through __call__: FSDP owns its parameter gathers.
            hidden, key, value = self.model.backbone.layers[0](
                torch.cat((embeds, features), -1), cached_step=cached_step
            )
            return self.model.backbone.final_norm(hidden), key, value

        if self.model.is_gradient_checkpointing and torch.is_grad_enabled():
            output, key, value = checkpoint(
                run, input_embeds, hidden_states, use_reentrant=False
            )
        else:
            output, key, value = run(input_embeds, hidden_states)
        self.keys, self.values = (*previous_keys, key), (*previous_values, value)
        return output


@dataclass(frozen=True)
class EagleCachedStep:
    """Immutable inputs for one recurrent stage, including its previous cache."""

    backend: str
    padding: torch.Tensor
    positions: torch.Tensor
    keys: tuple
    values: tuple
    rotary: object
    ring_group: object = None
    ulysses_group: object = None

    def forward(self, stage, hidden):
        embeds, features = hidden.split(stage.hidden_size, dim=-1)
        padding, positions, keys, values = (
            self.padding,
            self.positions,
            self.keys,
            self.values,
        )
        attention, node = stage.self_attn, stage.self_attn.config
        hidden = torch.cat(
            (stage.input_layernorm(embeds), stage.hidden_norm(features)), -1
        )
        batch, length = hidden.shape[:2]
        query, gate = attention.project_query(hidden)
        key = attention.k_norm(
            attention.k_proj(hidden).view(
                batch, length, node.num_key_value_heads, node.head_dim
            )
        ).transpose(1, 2)
        value = (
            attention.v_proj(hidden)
            .view(batch, length, node.num_key_value_heads, node.head_dim)
            .transpose(1, 2)
        )
        depth = len(keys)
        rotary = self.rotary
        if rotary is not None:
            rope_positions = positions + depth
            # Dynamic NTK must use the same global length on every SP rank.
            if self.backend == "usp":
                global_length = (
                    length
                    * dist.get_world_size(self.ring_group)
                    * dist.get_world_size(self.ulysses_group)
                    + depth
                )
                full = torch.arange(global_length, device=hidden.device)[None]
                cos, sin = rotary(hidden, full)
                embeddings = (cos[0, rope_positions], sin[0, rope_positions])
            else:
                embeddings = rotary(hidden, rope_positions)
            query, key = apply_rotary(query, embeddings), apply_rotary(key, embeddings)
        value = value.to(query.dtype)
        if self.backend in ("fa", "usp"):
            output, key, value = self._flash(
                query, key, value, keys, values, padding, depth
            )
            active = (
                torch.arange(length, device=padding.device)[None]
                < (padding.sum(-1) - depth)[:, None]
            )
        else:
            all_keys, all_values = (*keys, key), (*values, value)
            logical_positions = positions[0] if positions.ndim == 3 else positions
            key_positions = torch.cat(
                [logical_positions + i for i in range(depth + 1)], -1
            )
            valid_length = padding.sum(-1) - depth

            def visible(b, h, q, k):
                anchor = k % length
                return (
                    ((k < length) & (anchor <= q) | (anchor == q))
                    & (q < valid_length[b])
                    & (anchor < valid_length[b])
                )

            layout = AttentionLayout(
                query_positions=logical_positions + depth,
                key_positions=key_positions,
                visibility=visible,
                active_queries=torch.arange(length, device=padding.device)[None]
                < valid_length[:, None],
            )
            mask = prepare_attention_mask(
                layout,
                is_causal=node.is_causal,
                sliding_window=node.sliding_window,
                backend=self.backend,
                batch_size=batch,
                num_heads=node.num_attention_heads,
            )
            output, active = execute_attention(
                attention_backend(self.backend),
                query,
                torch.cat(all_keys, 2),
                torch.cat(all_values, 2),
                attention_mask=mask,
                scale=node.head_dim**-0.5,
                dropout_p=node.attention_dropout,
                training=stage.training,
            )
        output = attention.project_output(output, gate, active)
        residual = (
            stage.hidden_norm(features) if stage.norm_before_residual else features
        )
        hidden = residual + output
        hidden = hidden + stage.mlp(stage.post_attention_layernorm(hidden))
        return hidden, key, value

    def _flash(self, query, key, value, keys, values, padding, depth):
        from specforge.modeling.draft.layers.attention.backends.cached_flash import (
            _FlashCachedMergeFunc,
            _raise_standard_flash_attn_unavailable,
            _std_flash_attn_varlen_func,
            _USPRingFlashCachedMergeFunc,
        )

        if _std_flash_attn_varlen_func is None:
            _raise_standard_flash_attn_unavailable()
        query, key, value = [tensor.transpose(1, 2) for tensor in (query, key, value)]
        ring = ulysses = None
        if self.backend == "usp":
            from yunchang.comm import SeqAllToAll4D

            ring, ulysses = self.ring_group, self.ulysses_group
            if ring is None or ulysses is None:
                raise ValueError(
                    "USP requires explicit ring and Ulysses process groups"
                )
            query, key, value = [
                SeqAllToAll4D.apply(ulysses, tensor, 2, 1, False)
                for tensor in (query, key, value)
            ]
        all_keys = torch.stack((*keys, key), 1)
        all_values = torch.stack((*values, value), 1)
        scale = query.shape[-1] ** -0.5
        if ring is not None and dist.get_world_size(ring) > 1:
            output = _USPRingFlashCachedMergeFunc.apply(
                query, all_keys, all_values, ring, scale
            )
        else:
            lengths = (
                (padding.sum(-1) - depth).clamp(0, query.shape[1])
                if ulysses is None
                else torch.full(
                    (query.shape[0],),
                    query.shape[1],
                    device=query.device,
                    dtype=torch.long,
                )
            )
            output = _FlashCachedMergeFunc.apply(
                query, all_keys, all_values, lengths, scale
            )
        if ulysses is not None:
            output = SeqAllToAll4D.apply(ulysses, output, 1, 2, False)
        return output.transpose(1, 2), key, value
