import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import require_mlp_sync, require_mlp_tp_gather
from transformers import AutoModelForCausalLM

from specforge.distributed import (
    get_draft_tp_size,
    get_target_tp_device_mesh,
    get_target_tp_group,
    get_target_tp_rank,
    get_target_tp_size,
)
from specforge.utils import padding

from ...data import DataCollatorWithPadding
from .sglang_backend import SGLangRunner


def get_dp_data_shard_from_tp(
    tensor: Union[torch.Tensor, List[torch.Tensor]]
) -> torch.Tensor:
    """
    Get the data shard from the tensor.
    """
    target_tp_size = get_target_tp_size()
    target_tp_rank = get_target_tp_rank()
    draft_tp_size = get_draft_tp_size()
    if target_tp_size <= draft_tp_size:
        return tensor

    tensor_length = len(tensor) if isinstance(tensor, List) else tensor.shape[0]
    assert (
        tensor_length % target_tp_size == 0
    ), "Tensor length must be divisible by target_tp_size"
    chunk_size = tensor_length // (target_tp_size // draft_tp_size)
    return tensor[
        (target_tp_rank // draft_tp_size)
        * chunk_size : (target_tp_rank // draft_tp_size + 1)
        * chunk_size
    ]


@dataclass
class Eagle3TargetOutput:
    hidden_states: torch.Tensor  # [B, S, H*3]
    target: torch.Tensor  # [B, S, H]
    loss_mask: torch.Tensor  # [B, S, 1]
    input_ids: torch.Tensor  # [B, S]
    attention_mask: torch.Tensor  # [B, S]


class Eagle3TargetModel(ABC):
    """
    This  offers a layer of abstraction for the target model backend. The user can choose different backends to suit their needs:
    1. SGLang backend: for the mainstream model support with the fastest inference speed
    2. HuggingFace backend: for models that are not supported by SGLang but can be loaded by HuggingFace.
    3. Custom backend: for models with customized architecture and inference plan.
    """

    def __init__(self):
        self.aux_hidden_states_layers = None

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "Eagle3TargetModel":
        """
        Initialize the target model backend from a pretrained model path.
        """
        pass

    @torch.no_grad()
    def generate_eagle3_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> List[Eagle3TargetOutput]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # extract the aux hidden states
        # output_hidden_states = True will return the embedding output as well
        # so we have an offset of 1
        offset = 1
        low_aux_layer = self.aux_hidden_states_layers[0] + offset
        mid_aux_layer = self.aux_hidden_states_layers[1] + offset
        last_aux_layer = self.aux_hidden_states_layers[2] + offset

        hidden_states0 = outputs.hidden_states[low_aux_layer]
        hidden_states1 = outputs.hidden_states[mid_aux_layer]
        hidden_states2 = outputs.hidden_states[last_aux_layer]

        hidden_states = torch.cat(
            (hidden_states0, hidden_states1, hidden_states2), dim=-1
        )

        # apply pading
        target = outputs.logits
        target = padding(target, left=False)
        input_ids = padding(input_ids, left=False)
        loss_mask = loss_mask[..., None]
        loss_mask = loss_mask.to(target.device)

        return [
            Eagle3TargetOutput(
                hidden_states=get_dp_data_shard_from_tp(hidden_states),
                target=get_dp_data_shard_from_tp(target),
                loss_mask=get_dp_data_shard_from_tp(loss_mask),
                input_ids=get_dp_data_shard_from_tp(input_ids),
                attention_mask=get_dp_data_shard_from_tp(attention_mask),
            )
        ]

    def set_aux_hidden_states_layers(
        self, aux_hidden_states_layers: Optional[List[int]] = None
    ) -> None:
        """
        Set the layers to capture the aux hidden states from the target model outputs.
        """
        if aux_hidden_states_layers is None:
            if hasattr(self.model.config, "num_hidden_layers"):
                num_layers = self.model.config.num_hidden_layers
            else:
                raise ValueError(
                    f"Failed to set aux hidden states layers as model config {self.model.config} does not have num_hidden_layers"
                )
            aux_hidden_states_layers = [
                1,
                num_layers // 2 - 1,
                num_layers - 4,
            ]
        self.aux_hidden_states_layers = aux_hidden_states_layers
        assert (
            len(self.aux_hidden_states_layers) == 3
        ), "aux_hidden_states_layers is expected to be 3 layers for EAGLE3"


class HFEagle3TargetModel(Eagle3TargetModel):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "HFEagle3TargetModel":
        """
        Initialize the HuggingFace target model backend from a pretrained model path.
        """
        tp_size = get_target_tp_group().size()

        if tp_size > 1:
            device_kwargs = {
                "tp_plan": "auto",
                "tp_size": tp_size,
                "device_mesh": get_target_tp_device_mesh(),
            }
        else:
            device_kwargs = {
                "device_map": device,
            }

        target_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            **device_kwargs,
            **kwargs,
        )
        return cls(target_model)


class SGLangEagle3TargetModel(Eagle3TargetModel):

    def __init__(self, model_runner: SGLangRunner, target_micro_batch_size: int):
        super().__init__()
        self.model_runner = model_runner
        self.draft_data_collator = DataCollatorWithPadding()
        self.target_micro_batch_size = target_micro_batch_size

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        args: Optional[argparse.Namespace] = None,
        **kwargs,
    ) -> "SGLangEagle3TargetModel":
        server_args = ServerArgs.from_cli_args(args)
        server_args.enable_return_hidden_states = True
        model_config = ModelConfig.from_server_args(server_args)
        tp_rank = get_target_tp_rank()
        model_runner = SGLangRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=torch.cuda.current_device(),
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            moe_ep_rank=tp_rank // (server_args.tp_size // server_args.ep_size),
            moe_ep_size=server_args.ep_size,
            pp_rank=0,
            pp_size=1,
            server_args=server_args,
            nccl_port=None,
        )
        return cls(model_runner, args.target_micro_batch_size)

    def set_aux_hidden_states_layers(
        self, aux_hidden_states_layers: Optional[List[int]] = None
    ) -> None:
        self.model_runner.model.set_eagle3_layers_to_capture(aux_hidden_states_layers)

    @torch.no_grad
    def extend(self, reqs, capture_aux_hidden_states: bool = True):
        # Create dummy tree_cache for benchmarks (no prefix caching, just allocation)
        tree_cache = RadixCache(
            None,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            page_size=self.model_runner.server_args.page_size,
        )

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        self._maybe_prepare_mlp_sync_batch(batch)
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        logits_output, _ = self.model_runner.forward(forward_batch)

        aux_hidden_states_list = None
        input_lens = [len(req.origin_input_ids) for req in reqs]
        hidden_states = torch.split(logits_output.hidden_states, input_lens, dim=0)
        if capture_aux_hidden_states:
            aux_hidden_states_list = torch.split(
                logits_output.aux_hidden_states, input_lens, dim=0
            )
        else:
            aux_hidden_states_list = None

        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()
        return hidden_states, aux_hidden_states_list

    def _maybe_prepare_mlp_sync_batch(self, batch: ScheduleBatch):
        if require_mlp_sync(self.model_runner.server_args):
            Scheduler.prepare_mlp_sync_batch_raw(
                batch,
                dp_size=self.model_runner.server_args.dp_size,
                attn_tp_size=1,
                tp_group=self.model_runner.tp_group,
                get_idle_batch=None,
                disable_cuda_graph=self.model_runner.server_args.disable_cuda_graph,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                speculative_num_draft_tokens=None,
                require_mlp_tp_gather=require_mlp_tp_gather(
                    self.model_runner.server_args
                ),
                disable_overlap_schedule=self.model_runner.server_args.disable_overlap_schedule,
                offload_tags=set(),
            )

    @torch.no_grad()
    def generate_eagle3_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> List[Eagle3TargetOutput]:
        """
        args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            loss_mask: (batch_size, seq_len)
        return:
            data_for_draft: List[Dict[str, torch.Tensor]] of draft_batch_size, draft_micro_batch_size = 1
                - input_ids: (1, seq_len)
                - attention_mask: (1, seq_len)
                - loss_mask: (1, seq_len)
                - target: (1, seq_len, vocab_size) or (1, seq_len, hidden_size)
                - hidden_states: (1, seq_len, hidden_size)
        """
        sampling_params = SamplingParams(temperature=0, max_new_tokens=1, top_k=1)
        reqs, data_cache = [], []
        data_for_draft = []
        padding_len = (
            input_ids.shape[0]
            - input_ids.shape[0]
            // self.target_micro_batch_size
            * self.target_micro_batch_size
        )
        if padding_len > 0:
            input_ids = torch.cat(
                [
                    input_ids,
                    torch.zeros(
                        (padding_len, *input_ids.shape[1:]),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    ),
                ]
            )
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.zeros(
                        (padding_len, *attention_mask.shape[1:]),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    ),
                ]
            )
            loss_mask = torch.cat(
                [
                    loss_mask,
                    torch.zeros(
                        (padding_len, *loss_mask.shape[1:]),
                        device=loss_mask.device,
                        dtype=loss_mask.dtype,
                    ),
                ]
            )
        for idx, (input_id_, attention_mask_, loss_mask_) in enumerate(
            zip(
                input_ids.unsqueeze(1),
                attention_mask.unsqueeze(1),
                loss_mask.unsqueeze(1),
            )
        ):
            req = Req(
                rid=str(idx),
                origin_input_text="",
                origin_input_ids=input_id_.view(-1).tolist(),
                sampling_params=sampling_params,
            )
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
            req.logprob_start_len = len(req.origin_input_ids) - 1
            data_cache.append([input_id_, attention_mask_, loss_mask_])
            reqs.append(req)
            if len(reqs) == self.target_micro_batch_size:
                hidden_states_list, aux_hidden_states_list = self.extend(
                    reqs, capture_aux_hidden_states=True
                )
                hidden_states_list = get_dp_data_shard_from_tp(list(hidden_states_list))
                aux_hidden_states_list = get_dp_data_shard_from_tp(
                    list(aux_hidden_states_list)
                )
                data_cache = get_dp_data_shard_from_tp(list(data_cache))
                for data, hidden_states, aux_hidden_states in zip(
                    data_cache, hidden_states_list, aux_hidden_states_list
                ):
                    data_for_draft.append(
                        Eagle3TargetOutput(
                            hidden_states=aux_hidden_states.unsqueeze(0),
                            target=padding(hidden_states.unsqueeze(0), left=False),
                            loss_mask=data[2].unsqueeze(-1),
                            input_ids=padding(data[0], left=False),
                            attention_mask=data[1],
                        )
                    )
                reqs, data_cache = [], []
        return data_for_draft


class CustomEagle3TargetModel(Eagle3TargetModel):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "CustomEagle3TargetModel":
        from specforge.modeling.target import AutoDistributedTargetModel

        target_model = AutoDistributedTargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            device=device,
            **kwargs,
        )
        return cls(target_model)


def get_eagle3_target_model(
    pretrained_model_name_or_path: str,
    backend: str = "sglang",
    torch_dtype: torch.dtype = None,
    device: str = None,
    cache_dir: Optional[str] = None,
    args: Optional[argparse.Namespace] = None,
    **kwargs,
) -> Eagle3TargetModel:
    if backend == "sglang":
        return SGLangEagle3TargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            args=args,
            **kwargs,
        )
    elif backend == "hf":
        return HFEagle3TargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    elif backend == "custom":
        return CustomEagle3TargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")
