import gc
import glob
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Generator, Optional

import safetensors.torch
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from safetensors import safe_open
from tqdm import tqdm

from specforge.layers.linear import ColumnParallelLinear, RowParallelLinear


class DistributedTargetModel(ABC):

    def _load_ckpt_files(
        self, model_path: str, cache_dir: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Load the embedding of the draft model.

        Args:
            model_path (str): The path to the huggingface repository.
        """
        if os.path.exists(model_path):
            # model_path is a local directory
            # check if there is file ending with index.json
            glob_path = os.path.join(model_path, "*.index.json")
            index_json_path = glob.glob(glob_path)

            if len(index_json_path) > 1:
                raise FileNotFoundError(
                    f"Multiple index.json files found in {model_path}"
                )
            elif len(index_json_path) == 0:
                # there is no index.json file
                # so there are only two files supported:
                # model.safetensors, pytorch_model.bin
                # we need to check if there is only one of them
                safetensors_path = os.path.join(model_path, "model.safetensors")
                pytorch_model_path = os.path.join(model_path, "pytorch_model.bin")
                if os.path.exists(safetensors_path) and os.path.exists(
                    pytorch_model_path
                ):
                    raise FileNotFoundError(
                        f"Multiple model files found in {model_path}"
                    )
                elif os.path.exists(safetensors_path):
                    yield safetensors_path
                elif os.path.exists(pytorch_model_path):
                    yield pytorch_model_path
                else:
                    raise FileNotFoundError(f"No model file found in {model_path}")
            else:
                index_json_path = index_json_path[0]
                with open(index_json_path, "r") as f:
                    index_json = json.load(f)
                weight_map = index_json["weight_map"]
                ckpt_files = set(weight_map.values())
                for ckpt_file in ckpt_files:
                    yield os.path.join(model_path, ckpt_file)
        else:
            # this is the case where model_path is a huggingface repository
            # we first need to locate its local cache
            local_cache_path = snapshot_download(
                repo_id=model_path, cache_dir=cache_dir
            )
            yield from self._load_ckpt_files(local_cache_path)

    def _open_ckpt_file(self, ckpt_file: str) -> Dict[str, torch.Tensor]:
        if ckpt_file.endswith(".safetensors"):
            with safe_open(os.path.join(ckpt_file), framework="pt") as f:
                state_dict = {key: f.get_tensor(key) for key in f.keys()}
                return state_dict
        else:
            state_dict = torch.load(os.path.join(ckpt_file))
            return state_dict

    def _shard_tensor(self, tensor, process_group=None, dim=-1):
        rank = dist.get_rank(process_group)
        size = dist.get_world_size(process_group)
        return tensor.chunk(size, dim=dim)[rank].contiguous()

    def _gather_tensor(self, tensor, process_group=None, dim=-1):
        size = dist.get_world_size(process_group)
        obj_list = [torch.empty_like(tensor) for _ in range(size)]
        dist.all_gather(obj_list, tensor, group=process_group)
        gather_tensor = torch.cat(obj_list, dim=dim)
        return gather_tensor

    def load_checkpoint(self, checkpoint_path: str, cache_dir: Optional[str] = None):
        for ckpt_file in self._load_ckpt_files(checkpoint_path, cache_dir):
            state_dict = self._open_ckpt_file(ckpt_file)
            self.load_weights(state_dict)

    @abstractmethod
    def load_weights(self, state_dict: Dict[str, torch.Tensor]):
        """
        Load the weights of the target model.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state dict of the target model.
        """
        pass
