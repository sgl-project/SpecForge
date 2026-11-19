"""Small independent normalization/rotation formulas used by component tests."""

import torch
from torch import nn


class OffsetRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (value * (1 + self.weight.float())).to(x.dtype)


class PartialRotaryEmbedding(nn.Module):
    def __init__(self, config, head_dim):
        super().__init__()
        dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
        inv = 1.0 / (
            getattr(config, "rope_theta", 10000.0)
            ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, hidden, positions):
        phase = (
            self.inv_freq[None, :, None].float() @ positions[:, None, :].float()
        ).transpose(1, 2)
        phase = torch.cat((phase, phase), -1)
        return phase.cos().to(hidden.dtype), phase.sin().to(hidden.dtype)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    width = cos.shape[-1]

    def rotate(x):
        active, tail = x[..., :width], x[..., width:]
        first, second = active.chunk(2, dim=-1)
        return torch.cat(
            (active * cos + torch.cat((-second, first), -1) * sin, tail), -1
        )

    return rotate(q), rotate(k)
