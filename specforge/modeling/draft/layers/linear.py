import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from specforge.distributed import get_tp_group, shard_tensor


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        kv_head_replicas=False,
        layout_type: str = "normal",
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.layout_type = layout_type
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

        self._register_load_state_dict_pre_hook(self.shard_state_dict)

    def shard_state_dict(self, state_dict, *args):
        """
        This is a state dict hook to be triggered before loading the state dict. This will shard the weights and biases according to the layout type.
        """
        if self.layout_type == "normal":
            self.handle_normal_layout(state_dict, *args)
        else:
            raise ValueError(f"Invalid layout type: {self.layout_type}")

    def handle_normal_layout(self, state_dict, *args):
        # shard the weights
        if "weight" in state_dict:
            state_dict["weight"] = shard_tensor(state_dict["weight"], self.tp_group, -1)

        if "bias" in state_dict and self.tp_rank != 0:
            state_dict["bias"] = torch.zeros_like(state_dict["bias"])

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

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
        layout_type: str = "normal",
        kv_head_replicas=False,
        kv_head_idx=None,
        total_num_kv_heads=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.layout_type = layout_type
        self.tp_group = get_tp_group()
        self.tp_size = dist.get_world_size(self.tp_group)
        self.tp_rank = dist.get_rank(self.tp_group)

        self.in_features = in_features
        self.out_features = out_features
        self.kv_head_replicas = kv_head_replicas
        self.kv_head_idx = kv_head_idx
        self.total_num_kv_heads = total_num_kv_heads
        if self.kv_head_replicas:
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

        self._register_load_state_dict_pre_hook(self.shard_state_dict)

    def shard_state_dict(self, state_dict, *args):
        """
        This is a state dict hook to be triggered before loading the state dict. This will shard the weights and biases according to the layout type.
        """
        if self.kv_head_replicas:
            assert self.kv_head_idx is not None
            assert self.layout_type == "normal"
            self.handle_kv_head_replicas(state_dict, *args)
        else:
            if self.layout_type == "normal":
                self.handle_normal_layout(state_dict, *args)
            elif self.layout_type == "merged_qkv":
                self.handle_merged_qkv(state_dict, *args)
            elif self.layout_type == "gate_up":
                self.handle_gate_up_layout(state_dict, *args)
            else:
                raise ValueError(f"Invalid layout type: {self.layout_type}")

    def handle_kv_head_replicas(self, state_dict, *args):
        """
        This is a special case for GQA where the key/value are split according to the number of kv heads and the head which belongs to this rank.
        As the TP size is larger than the number of kv heads, we only keep one kv head per rank.
        """
        if "weight" in state_dict:
            state_dict["weight"] = state_dict["weight"].chunk(
                self.total_num_kv_heads, dim=0
            )[self.kv_head_idx]
        if "bias" in state_dict and state_dict["bias"] is not None:
            state_dict["bias"] = state_dict["bias"].chunk(
                self.total_num_kv_heads, dim=0
            )[self.kv_head_idx]

    def handle_normal_layout(self, state_dict, *args):
        """
        This shards the weights and biases along the column dimension.
        """
        # shard the weights
        if "weight" in state_dict:
            state_dict["weight"] = shard_tensor(state_dict["weight"], self.tp_group, 0)

        if "bias" in state_dict and state_dict["bias"] is not None:
            state_dict["bias"] = shard_tensor(state_dict["bias"], self.tp_group, 0)

    def handle_gate_up_layout(self, state_dict, *args):
        """
        This handles the gate_up layout where the gate and up weights are concatenated along the column dimension.
        """
        if "weight" in state_dict:
            gate, up = state_dict["weight"].chunk(2, dim=0)
            gate = shard_tensor(gate, self.tp_group, 0)
            up = shard_tensor(up, self.tp_group, 0)
            state_dict["weight"] = torch.cat((gate, up), dim=0)

        if "bias" in state_dict and state_dict["bias"] is not None:
            gate, up = state_dict["bias"].chunk(2, dim=0)
            gate = shard_tensor(gate, self.tp_group, 0)
            up = shard_tensor(up, self.tp_group, 0)
            state_dict["bias"] = torch.cat((gate, up), dim=0)

    def handle_merged_qkv(self, state_dict, *args):
        """
        This handles the merged QKV layout where the q, k, v weights are concatenated along the column dimension.
        """
        if "weight" in state_dict:
            # need to split into qkv and take the correct chunk for the rank
            q, k, v = state_dict["weight"].chunk(3, dim=0)
            q = shard_tensor(q, self.tp_group, 0)
            k = shard_tensor(k, self.tp_group, 0)
            v = shard_tensor(v, self.tp_group, 0)
            state_dict["weight"] = torch.cat((q, k, v), dim=0)

        if "bias" in state_dict and state_dict["bias"] is not None:
            q, k, v = state_dict["bias"].chunk(3, dim=0)
            q = shard_tensor(q, self.tp_group, 0)
            k = shard_tensor(k, self.tp_group, 0)
            v = shard_tensor(v, self.tp_group, 0)
            state_dict["bias"] = torch.cat((q, k, v), dim=0)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def __repr__(self):
        return f"ColumnParallelLinear(in_features={self.in_features}, out_features={self.out_features_per_shard}, tp_size={self.tp_size}, tp_rank={self.tp_rank})"
