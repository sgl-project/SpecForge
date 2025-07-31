import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from specforge.distributed import get_tp_group


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        kv_head_replicas=False,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.tp_group = get_tp_group()
        self.tp_size = dist.get_world_size(self.tp_group)
        self.tp_rank = dist.get_rank(self.tp_group)

        self.in_features = in_features
        self.out_features = out_features

        if kv_head_replicas:
            self.in_features_per_shard = in_features
        else:
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
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        kv_head_replicas=False,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.tp_group = get_tp_group()
        self.tp_size = dist.get_world_size(self.tp_group)
        self.tp_rank = dist.get_rank(self.tp_group)

        self.in_features = in_features
        self.out_features = out_features
        if kv_head_replicas:
            self.out_features_per_shard = out_features
        else:
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
