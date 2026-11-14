"""Direct tensor formulas for attention and dynamic convolution comparisons."""

import torch
import torch.nn.functional as F

from tests.test_modeling.rotary_reference import apply_mla_rope


def attend(q, k, v, mask, scale, dropout):
    k = k.repeat_interleave(q.shape[1] // k.shape[1], 1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], 1)
    scores = (q @ k.transpose(-1, -2)) * scale
    visible = mask if mask.dtype == torch.bool else torch.isfinite(mask)
    active = visible.any(-1, keepdim=True)
    scores = scores.masked_fill(~visible, -torch.inf).masked_fill(~active, 0)
    probabilities = scores.softmax(-1, dtype=torch.float32).to(q.dtype)
    probabilities = F.dropout(probabilities, p=dropout, training=True)
    return (probabilities @ v.to(q.dtype)).masked_fill(~active, 0)


def context_gqa(module, hidden, context, embeddings, mask):
    batch, length, _ = hidden.shape
    width = module.head_dim
    q = module.q_norm(module.q_proj(hidden).view(batch, length, -1, width)).transpose(
        1, 2
    )
    keys, values = [], []
    for source in (context, hidden):
        keys.append(module.k_proj(source))
        values.append(
            module.v_proj(source)
            .view(batch, source.shape[1], -1, width)
            .transpose(1, 2)
        )
    cos, sin = embeddings
    q = apply_mla_rope(q, cos[:, -length:], sin[:, -length:], interleaved=False)
    key = torch.cat(keys, 1)
    key = module.k_norm(key.view(batch, key.shape[1], -1, width)).transpose(1, 2)
    k = apply_mla_rope(key, cos, sin, interleaved=False)
    output = attend(
        q, k, torch.cat(values, 2), mask, width**-0.5, module.attention_dropout
    )
    return module.o_proj(output.transpose(1, 2).reshape(batch, length, -1))


def context_mla(module, hidden, context, embeddings, mask):
    node = module.config
    batch, length, _ = hidden.shape
    q = (
        module.q_proj(hidden)
        if node.q_lora_rank is None
        else module.q_b_proj(module.q_a_layernorm(module.q_a_proj(hidden)))
    )
    q = q.view(batch, length, node.num_attention_heads, node.qk_head_dim).transpose(
        1, 2
    )
    q_plain, q_rotary = q.split((node.qk_nope_head_dim, node.qk_rope_head_dim), -1)
    keys, rotations, values = [], [], []
    for source in (context, hidden):
        latent, rotation = module.kv_a_proj_with_mqa(source).split(
            (node.kv_lora_rank, node.qk_rope_head_dim), -1
        )
        projected = (
            module.kv_b_proj(module.kv_a_layernorm(latent))
            .view(batch, source.shape[1], node.num_attention_heads, -1)
            .transpose(1, 2)
        )
        key, value = projected.split((node.qk_nope_head_dim, node.v_head_dim), -1)
        keys.append(key)
        values.append(value)
        rotations.append(rotation.unsqueeze(1))
    cos, sin = embeddings
    q_rotary = apply_mla_rope(
        q_rotary,
        cos[:, -length:],
        sin[:, -length:],
        interleaved=module.rotary_dim > 0 and module.rope.interleaved,
    )
    k_rotary = apply_mla_rope(
        torch.cat(rotations, 2),
        cos,
        sin,
        interleaved=module.rotary_dim > 0 and module.rope.interleaved,
    )
    q = torch.cat((q_plain, q_rotary), -1)
    k = torch.cat(
        (torch.cat(keys, 2), k_rotary.expand(-1, node.num_attention_heads, -1, -1)), -1
    )
    output = attend(
        q, k, torch.cat(values, 2), mask, module.scaling, node.attention_dropout
    )
    output = module.o_proj(output.transpose(1, 2).reshape(batch, length, -1))
    return output.masked_fill(~mask.any(-1).squeeze(1).unsqueeze(-1), 0)


def convolution_pair(module, hidden):
    batch, length, width = hidden.shape
    taps, groups = module.config.kernel_size, module.num_groups
    coefficients = module.kernel_projection(hidden).reshape(
        batch, length, 2, taps, groups
    )

    def convolve(values, side):
        blocks = values.reshape(
            batch, -1, module.block_size, groups, module.config.group_size
        )
        weights = coefficients[:, :, side].reshape(
            batch, -1, module.block_size, taps, groups, 1
        )
        weights = weights + module.base_kernel[side].reshape(1, 1, 1, taps, groups, -1)
        output = weights[:, :, :, 0] * blocks
        for tap in range(1, taps):
            output = output + weights[:, :, :, tap] * F.pad(
                blocks[:, :, :-tap], (0, 0, 0, 0, tap, 0)
            )
        return output.reshape(batch, length, width)

    return convolve(torch.tanh(convolve(hidden, 0)), 1)
