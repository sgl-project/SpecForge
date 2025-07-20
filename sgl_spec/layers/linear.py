import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from sgl_spec.distributed import get_tp_group


class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.tp_group = get_tp_group()
        self.tp_size = dist.get_world_size(self.tp_group)
        self.tp_rank = dist.get_rank(self.tp_group)

        self.in_features = in_features
        self.out_features = out_features
        self.in_features_per_shard = in_features // self.tp_size
        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features_per_shard, **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def load_state_dict(self, state_dict, strict=True):
        weight_of_current_shard = state_dict["weight"][
            self.tp_rank
            * self.in_features_per_shard : (self.tp_rank + 1)
            * self.in_features_per_shard
        ]
        self.weight.data.copy_(weight_of_current_shard)
        if self.bias is not None:
            bias_of_current_shard = state_dict["bias"][
                self.tp_rank
                * self.in_features_per_shard : (self.tp_rank + 1)
                * self.in_features_per_shard
            ]
            self.bias.data.copy_(bias_of_current_shard)

    def __repr__(self):
        return f"RowParallelLinear(in_features={self.in_features_per_shard}, out_features={self.out_features}, tp_size={self.tp_size}, tp_rank={self.tp_rank})"


class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.tp_group = get_tp_group()
        self.tp_size = dist.get_world_size(self.tp_group)
        self.tp_rank = dist.get_rank(self.tp_group)

        self.in_features = in_features
        self.out_features = out_features
        self.out_features_per_shard = out_features // self.tp_size

        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_shard, self.in_features, **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features_per_shard, **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def load_state_dict(self, state_dict, strict=True):
        weight_of_current_shard = state_dict["weight"][
            self.tp_rank
            * self.out_features_per_shard : (self.tp_rank + 1)
            * self.out_features_per_shard,
            :,
        ]
        self.weight.data.copy_(weight_of_current_shard)
        if self.bias is not None:
            bias_of_current_shard = state_dict["bias"][
                self.tp_rank
                * self.out_features_per_shard : (self.tp_rank + 1)
                * self.out_features_per_shard
            ]
            self.bias.data.copy_(bias_of_current_shard)

    def __repr__(self):
        return f"ColumnParallelLinear(in_features={self.in_features}, out_features={self.out_features_per_shard}, tp_size={self.tp_size}, tp_rank={self.tp_rank})"


class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        config=None,
        padding_idx=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.tp_group = get_tp_group()
        self.tp_size = dist.get_world_size(self.tp_group)
        self.tp_rank = dist.get_rank(self.tp_group)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.vocab_start = vocab_size * self.tp_rank // self.tp_size
        self.vocab_end = vocab_size * (self.tp_rank + 1) // self.tp_size
        self.vocab_size_per_shard = self.vocab_end - self.vocab_start

        if (
            padding_idx is None
            and config is not None
            and hasattr(config, "pad_token_id")
        ):
            padding_idx = config.pad_token_id
        self.padding_idx = padding_idx

        self.weight = nn.Parameter(
            torch.empty(self.vocab_size_per_shard, embedding_dim, **factory_kwargs)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim**-0.5)
        if self.padding_idx is not None:
            if self.vocab_start <= self.padding_idx < self.vocab_end:
                local_idx = self.padding_idx - self.vocab_start
                with torch.no_grad():
                    self.weight[local_idx].zero_()

    def forward(self, input):
        mask = (input >= self.vocab_start) & (input < self.vocab_end)
        local_input = input.clone() - self.vocab_start
        local_input[~mask] = 0

        output = torch.zeros(
            *input.shape,
            self.embedding_dim,
            device=input.device,
            dtype=self.weight.dtype,
        )
        output[mask] = self.weight[local_input[mask]]

        if self.padding_idx is not None:
            output[input == self.padding_idx] = 0

        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.tp_group)
        return output

    def extra_repr(self):
        return f"vocab_size={self.vocab_size}, embedding_dim={self.embedding_dim}, tp_size={self.tp_size}, tp_rank={self.tp_rank}, padding_idx={self.padding_idx}"

    def load_state_dict(self, state_dict, strict=True):
        # 只加载属于本shard的embedding参数
        weight_of_current_shard = state_dict["weight"][
            self.vocab_start : self.vocab_end, :
        ]
        self.weight.data.copy_(weight_of_current_shard)
