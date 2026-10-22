"""Building blocks of the LongSpark drafter."""

import torch
from torch import nn
from torch.nn import functional as F

from .longspark_context import TargetContext


class FeedForward(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        bias: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=bias),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features, bias=bias),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + self.eps)
        return normalized.to(dtype=hidden_states.dtype) * self.weight.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )


class DraftStateEncoder(nn.Module):
    """Project the selected target hidden states into one draft state."""

    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()
        self.proj = FeedForward(num_layers * hidden_size, hidden_size, hidden_size)
        self.norm = RMSNorm(hidden_size)

    def forward(self, selected_layer_hidden: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(selected_layer_hidden.flatten(start_dim=1)))


class DraftBlockInput(nn.Module):
    """Build the ``[x_t, MASK, ..., MASK]`` draft block for each anchor."""

    def __init__(self, hidden_size: int, block_size: int, mask_token_id: int):
        super().__init__()
        self.block_size = int(block_size)
        self.mask_token_id = int(mask_token_id)
        self.position_embedding = nn.Embedding(block_size, hidden_size)
        self.proj = FeedForward(2 * hidden_size, hidden_size, hidden_size)
        self.norm = RMSNorm(hidden_size)

    def forward(
        self,
        *,
        current_token_ids: torch.Tensor,
        latest_target_state: torch.Tensor,
        target_embed_tokens: nn.Module,
        model_dtype: torch.dtype,
    ) -> torch.Tensor:
        device = latest_target_state.device
        token_ids = torch.full(
            (current_token_ids.shape[0], self.block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        token_ids[:, 0] = current_token_ids.to(device=device, dtype=torch.long)
        token_states = target_embed_tokens(token_ids).to(dtype=model_dtype)
        positions = torch.arange(self.block_size, device=device)
        token_states = (
            token_states
            + self.position_embedding(positions).to(dtype=model_dtype)[None, :, :]
        )
        latest = latest_target_state[:, None, :].expand(-1, self.block_size, -1)
        return self.norm(self.proj(torch.cat([token_states, latest], dim=-1)))


class LongSparkMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class LongSparkAttention(nn.Module):
    """Qwen3-style GQA over the target context and the draft block.

    Captured target keys already carry the target's K-norm and RoPE, so only
    the draft-block queries and keys are normalized and rotated here. Draft
    K/V is used within the block and never cached.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rope_theta: float,
        rms_norm_eps: float,
    ):
        super().__init__()
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.q_proj = nn.Linear(
            hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, hidden_size, bias=False
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        inv_freq = 1.0 / (
            float(rope_theta)
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotary_embeddings(
        self, position_ids: torch.Tensor, *, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = (
            position_ids.to(device=self.inv_freq.device, dtype=torch.float32)[..., None]
            * self.inv_freq[None, None, :]
        )
        embeddings = torch.cat([frequencies, frequencies], dim=-1)
        cos = embeddings.cos().to(dtype=dtype)[:, None, :, :]
        sin = embeddings.sin().to(dtype=dtype)[:, None, :, :]
        return cos, sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        context: TargetContext,
        layer_index: int,
    ) -> torch.Tensor:
        batch_size, block_size, _ = hidden_states.shape
        if position_ids.shape != (batch_size, block_size):
            raise ValueError("position_ids must have shape [anchors, block_size]")
        query = self.q_proj(hidden_states).view(
            batch_size, block_size, self.num_attention_heads, self.head_dim
        )
        block_key = self.k_proj(hidden_states).view(
            batch_size, block_size, self.num_key_value_heads, self.head_dim
        )
        block_value = self.v_proj(hidden_states).view(
            batch_size, block_size, self.num_key_value_heads, self.head_dim
        )
        query = self.q_norm(query).transpose(1, 2)
        block_key = self.k_norm(block_key).transpose(1, 2)
        block_value = block_value.transpose(1, 2)
        cos, sin = self._rotary_embeddings(position_ids, dtype=query.dtype)
        query = query * cos + _rotate_half(query) * sin
        block_key = block_key * cos + _rotate_half(block_key) * sin
        # The draft block attends bidirectionally within itself.
        block_attention_mask = torch.ones(
            (batch_size, block_size, block_size),
            dtype=torch.bool,
            device=hidden_states.device,
        )
        attended = context.attend(
            layer_index=layer_index,
            query=query,
            block_keys=block_key,
            block_values=block_value,
            block_attention_mask=block_attention_mask,
        )
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, block_size, self.num_attention_heads * self.head_dim)
        )
        return self.o_proj(attended)


class LongSparkDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rope_theta: float,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.self_attn = LongSparkAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
        )
        self.mlp = LongSparkMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        context: TargetContext,
        layer_index: int,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            position_ids=position_ids,
            context=context,
            layer_index=layer_index,
        )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
