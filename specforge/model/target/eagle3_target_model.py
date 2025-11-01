import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import require_mlp_sync, require_mlp_tp_gather
from transformers import AutoModelForCausalLM

from specforge.distributed import get_target_tp_device_mesh, get_target_tp_group
from specforge.utils import padding

from ...data import DataCollatorWithPadding
from .sglang_backend import SGLangRunner, wrap_eagle3_logits_processors_in_module
from .target_head import TargetHead


def get_dp_data_shard_from_tp(
    tensor: Union[torch.Tensor, List[torch.Tensor]]
) -> torch.Tensor:
    """
    Get the data shard from the tensor.
    """
    tp_size = dist.get_world_size(get_target_tp_group())
    tp_rank = dist.get_rank(get_target_tp_group())
    tensor_length = len(tensor) if isinstance(tensor, List) else tensor.shape[0]
    assert tensor_length % tp_size == 0, "Tensor length must be divisible by tp_size"
    chunk_size = tensor_length // tp_size
    return tensor[tp_rank * chunk_size : (tp_rank + 1) * chunk_size]


@dataclass
class Eagle3TargetOutput:
    hidden_states: torch.Tensor
    target: torch.Tensor
    loss_mask: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


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
    ) -> Eagle3TargetOutput:
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

        return Eagle3TargetOutput(
            hidden_states=get_dp_data_shard_from_tp(hidden_states),
            target=get_dp_data_shard_from_tp(target),
            loss_mask=get_dp_data_shard_from_tp(loss_mask),
            input_ids=get_dp_data_shard_from_tp(input_ids),
            attention_mask=get_dp_data_shard_from_tp(attention_mask),
        )

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
        self.target_head = None
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
        tp_rank = dist.get_rank(get_target_tp_group())
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
        wrap_eagle3_logits_processors_in_module(
            model_runner.model, return_full_logits=False
        )
        return cls(model_runner, args.target_micro_batch_size)

    def set_aux_hidden_states_layers(
        self, aux_hidden_states_layers: Optional[List[int]] = None
    ) -> None:
        self.model_runner.model.set_eagle3_layers_to_capture(aux_hidden_states_layers)

    @torch.no_grad
    def extend(self, reqs, capture_aux_hidden_states: bool = True):
        # Create dummy tree_cache for benchmarks (no prefix caching, just allocation)
        dummy_tree_cache = SimpleNamespace(
            page_size=self.model_runner.server_args.page_size,
            device=self.model_runner.device,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
        )

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=dummy_tree_cache,
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
        logits = torch.split(logits_output.logits, input_lens, dim=0)
        if capture_aux_hidden_states:
            aux_hidden_states_list = torch.split(
                logits_output.aux_hidden_states, input_lens, dim=0
            )
        else:
            aux_hidden_states_list = None

        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()
        return logits, aux_hidden_states_list

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
        input_ids: List[torch.Tensor],
        attention_mask: List[torch.Tensor],
        loss_mask: List[torch.Tensor],
    ) -> Eagle3TargetOutput:
        """
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
        for idx, (input_id_, attention_mask_, loss_mask_) in enumerate(
            zip(input_ids, attention_mask, loss_mask)
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
                logits_list, aux_hidden_states_list = self.extend(
                    reqs, capture_aux_hidden_states=True
                )
                logits_list = get_dp_data_shard_from_tp(list(logits_list))
                aux_hidden_states_list = get_dp_data_shard_from_tp(
                    list(aux_hidden_states_list)
                )
                data_cache = get_dp_data_shard_from_tp(list(data_cache))
                reqs, data_cache = [], []
                for data, logits, aux_hidden_states in zip(
                    data_cache, logits_list, aux_hidden_states_list
                ):
                    data_for_draft.append(
                        Eagle3TargetOutput(
                            hidden_states=aux_hidden_states.unsqueeze(0),
                            target=padding(logits.unsqueeze(0), left=False),
                            loss_mask=data[2],
                            input_ids=padding(data[0], left=False),
                            attention_mask=data[1],
                        )
                    )
        return data_for_draft

        # aux_hidden_states_out = []
        # target_out = []
        # loss_mask_out = []
        # input_ids_out = []

        # for idx, (data, logits, aux_hidden_states) in enumerate(
        #     zip(data_cache, logits_list, aux_hidden_states_list)
        # ):
        #     aux_hidden_states_out.append(aux_hidden_states.unsqueeze(0))
        #     target_out.append(logits.unsqueeze(0))
        #     loss_mask_out.append(data[2])
        #     input_ids_out.append(data[0])

        # aux_hidden_states_out = torch.cat(aux_hidden_states_out, dim=0)
        # target_out = torch.cat(target_out, dim=0)
        # loss_mask_out = torch.cat(loss_mask_out, dim=0)
        # input_ids_out = torch.cat(input_ids_out, dim=0)

        # target_out = padding(target_out, left=False)
        # input_ids_out = padding(input_ids_out, left=False)
        # loss_mask_out = loss_mask_out[..., None]

        # return Eagle3TargetOutput(
        #     hidden_states=aux_hidden_states_out,
        #     target=target_out,
        #     loss_mask=loss_mask_out,
        #     input_ids=input_ids_out,
        #     attention_mask=attention_mask,
        # )


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
        from specforge.modeling.auto import AutoDistributedTargetModel

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
    lm_head_key: str = "lm_head.weight",
    args: Optional[argparse.Namespace] = None,
    **kwargs,
) -> Eagle3TargetModel:
    if backend == "sglang":
        target_head = TargetHead(pretrained_model_name_or_path)
        target_head.load_weights(
            model_path=pretrained_model_name_or_path, lm_head_key=lm_head_key
        )
        target_head.freeze_weights()
        target_head = target_head.eval().cuda().to(torch_dtype)
        target_model = SGLangEagle3TargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            args=args,
            **kwargs,
        )
        target_model.target_head = target_head
        return target_model
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
