import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
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

class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        in_features,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias=True,
        device=None,
        dtype=None,
        kv_head_replicas=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        self.tp_group = get_tp_group()
        self.tp_size = dist.get_world_size(self.tp_group)
        self.tp_rank = dist.get_rank(self.tp_group)

        self.in_features = in_features
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.num_heads = self.total_num_heads // self.tp_size
        self.num_kv_heads = self.total_num_kv_heads // self.tp_size
        self.num_kv_head_replicas = 1
        self.q_proj_shard_size = self.num_heads * self.head_size
        self.kv_proj_shard_size = self.num_kv_heads * self.head_size
        input_size = self.in_features
        self.output_size = (
            (self.num_heads + 2 * self.num_kv_heads) * self.tp_size * self.head_size
        )
        self.output_sizes = [
            self.num_heads * self.head_size * self.tp_size,  # q_proj
            self.num_kv_heads * self.head_size * self.tp_size,  # k_proj
            self.num_kv_heads * self.head_size * self.tp_size,  # v_proj
        ]

        super().__init__(
            input_size,
            self.output_size,
            bias=bias,
            device=device,
            dtype=dtype,
            kv_head_replicas=kv_head_replicas,
        )

    def load_state_dict(self, state_dict, strict=True):
        full_weight = state_dict["weight"]
        weight_shape = full_weight.shape
        expected_out = (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_size
        assert weight_shape[0] == expected_out, (
            f"Expected full weight output dim {expected_out}, but got {weight_shape[0]}"
        )
        assert weight_shape[1] == self.in_features, (
            f"Expected input dim {self.in_features}, but got {weight_shape[1]}"
        )

        if self.kv_head_replicas:
            # Each rank holds the full output, but we still need to slice per-shard size?
            # Actually: out_features_per_shard == out_features when kv_head_replicas=True
            weight_of_current_shard = full_weight[: self.out_features_per_shard, :]
            if self.bias is not None:
                full_bias = state_dict["bias"]
                bias_of_current_shard = full_bias[: self.out_features_per_shard]
        else:
            shard_start = self.tp_rank * self.out_features_per_shard
            shard_end = shard_start + self.out_features_per_shard
            weight_of_current_shard = full_weight[shard_start:shard_end, :]
            if self.bias is not None:
                full_bias = state_dict["bias"]
                bias_of_current_shard = full_bias[shard_start:shard_end]

        self.weight.data.copy_(weight_of_current_shard)
        if self.bias is not None:
            self.bias.data.copy_(bias_of_current_shard)

    def __repr__(self):
        return f"QKVParallelLinear(in_features={self.in_features}, out_features={self.output_size}, tp_size={self.tp_size}, tp_rank={self.tp_rank})"
