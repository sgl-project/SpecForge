import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from specforge.distributed import get_tp_group, shard_tensor


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        if padding_idx is not None:
            if padding_idx > 0:
                assert (
                    padding_idx < self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert (
                    padding_idx >= -self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
                padding_idx = self.num_embeddings + padding_idx

        # tp-realted
        self.tp_group = get_tp_group()
        self.tp_rank = dist.get_rank(self.tp_group)
        self.tp_size = dist.get_world_size(self.tp_group)

        # deal with the case where the embedding is not divisible by the TP size
        self.num_embeddings_per_shard = math.ceil(num_embeddings / self.tp_size)
        self.padded_num_embeddings = (
            self.num_embeddings_per_shard * self.tp_size - self.num_embeddings
        )
        self.vocab_start_index = self.tp_rank * self.num_embeddings_per_shard
        self.vocab_end_index = min(
            self.vocab_start_index + self.num_embeddings_per_shard,
            self.num_embeddings,
        )

        if (
            padding_idx is not None
            and padding_idx >= self.vocab_start_index
            and padding_idx < self.vocab_end_index
        ):
            self.padding_idx = padding_idx - self.vocab_start_index
        else:
            self.padding_idx = None

        self.weight = nn.Parameter(
            torch.empty(
                (self.num_embeddings_per_shard, self.embedding_dim), **factory_kwargs
            ),
            requires_grad=True,
        )
        self.reset_parameters()
        self._register_load_state_dict_pre_hook(self.shard_state_dict)

    def shard_state_dict(self, state_dict, *args):
        if "weight" in state_dict:
            value = state_dict["weight"]

            # pad this value if it is not divisible by the TP size
            if value.shape[0] % self.tp_size != 0:
                padding_size = self.tp_size - value.shape[0] % self.tp_size
                value = F.pad(value, (0, 0, 0, padding_size))
            state_dict["weight"] = shard_tensor(value, self.tp_group, 0)

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def generate_mask(self, input_):
        # generate the mask for the vocab which is only owned by the current rank
        mask = (input_ >= self.vocab_start_index) & (input_ < self.vocab_end_index)
        return mask

    def forward(self, input_):
        if self.tp_size > 1:
            # Build the mask.
            mask = self.generate_mask(input_)
            masked_input = input_ - self.vocab_start_index
            masked_input[~mask] = 0
        else:
            masked_input = input_

        output_parallel = F.embedding(
            masked_input,
            self.weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

        # Mask the output embedding.
        if self.tp_size > 1:
            output_parallel[~mask] = 0
            # Reduce across all the model parallel GPUs.
            dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM, group=self.tp_group)
            output = output_parallel
        else:
            output = output_parallel
        return output
