import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from specforge.distributed import get_tp_group, shard_tensor


class ParallelLMHead(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.tp_group = get_tp_group()
        self.tp_size = dist.get_world_size(self.tp_group)
        self.tp_rank = dist.get_rank(self.tp_group)

        # tp-related
        self.out_features_per_shard = math.ceil(out_features / self.tp_size)
        self.padded_out_features = (
            self.out_features_per_shard * self.tp_size - out_features
        )
        assert (
            self.out_features_per_shard * self.tp_size
            == out_features + self.padded_out_features
        )

        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_shard, self.in_features, **factory_kwargs)
        )
        self.bias = (
            nn.Parameter(torch.zeros(self.out_features_per_shard, **factory_kwargs))
            if bias
            else None
        )

        # init params
        self.reset_parameters()

        # handle weight loading
        self._register_load_state_dict_pre_hook(self.shard_state_dict)

    def shard_state_dict(self, state_dict, *args):
        if "weight" in state_dict:
            value = state_dict["weight"]

            # pad this value if it is not divisible by the TP size
            if value.shape[0] % self.tp_size != 0:
                padding_size = self.tp_size - value.shape[0] % self.tp_size
                value = F.pad(value, (0, 0, 0, padding_size))
            state_dict["weight"] = shard_tensor(value, self.tp_group, 0)

        if "bias" in state_dict:
            value = state_dict["bias"]

            # pad this value if it is not divisible by the TP size
            if value.shape[0] % self.tp_size != 0:
                padding_size = self.tp_size - value.shape[0] % self.tp_size
                value = F.pad(value, (0, padding_size))
            state_dict["bias"] = shard_tensor(value, self.tp_group, 0)

    def forward(self, hidden: torch.Tensor, gather_output: bool = False):
        """
        hidden: [B, T, H] or [N, H]
        returns:
          - if gather_output=False: local logits [*, local_vocab] and (start,end) for stitching
          - if gather_output=True:  full logits [*, vocab] via all-gather (use for inference)
        """
        orig_shape = hidden.shape
        hidden = hidden.reshape(-1, self.in_features)  # [N, H]

        local_logits = hidden @ self.weight.T  # [N, local_vocab]

        if self.bias is not None:
            local_logits = local_logits + self.bias

        if not gather_output or self.tp_size == 1:
            return local_logits.view(
                *orig_shape[:-1], self.out_features_per_shard
            ).contiguous()
        else:
            # all-gather shards along vocab dim
            chunks = [torch.empty_like(local_logits) for _ in range(self.tp_size)]
            dist.all_gather(chunks, local_logits, group=self.tp_group)
            full = torch.cat(chunks, dim=-1)[
                :, : self.out_features
            ]  # trim padding from ceil-div
            return full.view(*orig_shape[:-1], self.out_features).contiguous()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def __repr__(self):
        return f"ParallelLMHead(in_features={self.in_features}, out_features={self.out_features_per_shard}, tp_size={self.tp_size}, tp_rank={self.tp_rank})"
